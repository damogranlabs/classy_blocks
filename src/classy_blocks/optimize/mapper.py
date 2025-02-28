from typing import List, Union

from classy_blocks.cbtyping import IndexType, NPPointType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL

ElementType = Union[Face, Operation]


class Mapper:
    """A helper that constructs mapped sketches/shapes
    from arbitrary collection of faces/operations"""

    def __init__(self) -> None:
        self.points: List[NPPointType] = []
        self.indexes: List[IndexType] = []
        self.elements: List[Union[Face, Operation]] = []

    def _add_point(self, point: NPPointType) -> int:
        # TODO: this code is repeated several times all over;
        # consolidate, unify, agglomerate, amass
        # (especially in case one would need to utilize an octree or something)
        for i, position in enumerate(self.points):
            if f.norm(point - position) < TOL:
                # reuse an existing point
                index = i
                break
        else:
            # no point found, create a new one
            index = len(self.points)
            self.points.append(point)

        return index

    def add(self, element: ElementType) -> None:
        """Add Face's or Operation's points to the map"""
        indexes = [self._add_point(point) for point in element.point_array]
        self.indexes.append(indexes)
        self.elements.append(element)

    @classmethod
    def from_map(cls, points: List[NPPointType], indexes: List[IndexType], elements: List[ElementType]) -> "Mapper":
        """Creates a ready-made mapper from a sketch/shape that already has points/indexes defined"""
        if len(indexes) != len(elements):
            raise ValueError("Number of indexes and elements don't match!")

        mapper = cls()
        mapper.points = points
        mapper.indexes = indexes
        mapper.elements = elements

        return mapper
