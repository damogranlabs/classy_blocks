import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.shapes.mapped.description import MappedShapeDescription


class MappedShape(Shape):
    """Similar to a MappedSketch, this object takes a list of points in space
    and their connectivity (list of indexes that form each hexahedron)
    to construct an arbitrary, user-defined Shape."""

    def __init__(self, description: MappedShapeDescription):
        self.description = description
        self.points = np.array(self.description.points)

        self._lofts: list[Loft] = []

        for hexa in self.description.hexas:
            bottom_face = Face(np.take(self.points, hexa[:4], axis=0))
            top_face = Face(np.take(self.points, hexa[4:], axis=0))

            self._lofts.append(Loft(bottom_face, top_face))

    @property
    def operations(self):
        return self._lofts

    @property
    def center(self):
        return np.average(self.points, axis=0)

    @property
    def grid(self):
        # Grid is irrelevant for arbitrary mapped shapes
        return [self.operations]

    @classmethod
    def from_file(cls, path: str) -> "MappedShape":
        description = MappedShapeDescription.load(path)
        return cls(description)
