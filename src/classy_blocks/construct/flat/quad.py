from typing import List, Set

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.types import NPPointListType, QuadIndexType
from classy_blocks.util import functions as f


class Quad:
    """A helper class for tracking positions-faces-indexes-neighbours-whatnot"""

    def __init__(self, positions: NPPointListType, indexes: QuadIndexType):
        self.indexes = indexes
        self.positions = positions
        self.face = Face([self.positions[i] for i in self.indexes])

    def update(self) -> None:
        """Update Face position"""
        self.face.update([self.positions[i] for i in self.indexes])

    @property
    def points(self) -> NPPointListType:
        return self.face.point_array

    @property
    def connections(self) -> List[Set[int]]:
        return [{self.indexes[i], self.indexes[(i + 1) % 4]} for i in range(4)]

    @property
    def perimeter(self):
        return sum([f.norm(self.points[i] - self.points[(i + 1) % 4]) for i in range(4)])

    @property
    def center(self):
        return np.average(self.points)

    @property
    def e1(self):
        return f.unit_vector(self.points[1] - self.points[0])

    @property
    def normal(self):
        return f.unit_vector(np.cross(self.points[1] - self.points[0], self.points[3] - self.points[0]))

    @property
    def e2(self):
        return f.unit_vector(-np.cross(self.e1, self.normal))

    @property
    def energy(self):
        e = 0

        ideal_side = self.perimeter / 2
        ideal_diagonal = 2**0.5 * ideal_side / 2
        center = self.center

        for i in range(4):
            e += (f.norm(self.points[i] - self.points[(i + 1) % 4]) - ideal_side) ** 2
            e += (f.norm(center - self.points[i]) - ideal_diagonal) ** 2

        return e
