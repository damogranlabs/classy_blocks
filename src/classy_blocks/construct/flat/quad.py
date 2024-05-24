from typing import List, Set

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.types import NPPointListType, NPPointType, QuadIndexType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class Quad:
    """A helper class for tracking positions-faces-indexes-neighbours-whatnot"""

    def __init__(self, positions: NPPointListType, indexes: QuadIndexType):
        self.indexes = indexes
        self.face = Face([positions[i] for i in self.indexes])

    def update(self, positions: NPPointListType) -> None:
        """Update Face position"""
        self.face.update([positions[i] for i in self.indexes])

    def contains(self, point: NPPointType) -> bool:
        """Returns True if the given point is a part of this quad"""
        for this_point in self.points:
            if f.norm(point - this_point) < TOL:
                return True

        return False

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
        return np.average(self.points, axis=0)

    @property
    def area(self):
        center = self.center
        sum_area = 0

        for i in range(4):
            side_1 = self.points[i] - center
            side_2 = self.points[(i + 1) % 4] - center

            sum_area += 0.5 * f.norm(np.cross(side_1, side_2))

        return sum_area

    @property
    def e1(self):
        return f.unit_vector(self.points[1] - self.points[0])

    @property
    def normal(self):
        return f.unit_vector(np.cross(self.points[1] - self.points[0], self.points[3] - self.points[0]))

    @property
    def e2(self):
        return f.unit_vector(-np.cross(self.e1, self.normal))
