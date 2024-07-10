from typing import List

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.types import IndexType, NPPointListType, NPPointType, PointListType
from classy_blocks.util import functions as f


class MappedSketch(Sketch):
    """A sketch that is created from predefined points.
    The points are connected to form quads which define Faces."""

    def __init__(self, positions: PointListType, quads: List[IndexType]):
        self._faces: List[Face] = []
        self.indexes = quads

        for quad in self.indexes:
            face = Face([positions[iq] for iq in quad])
            self._faces.append(face)

        self.add_edges()

    def add_edges(self) -> None:
        """An optional method that will add edges to faces;
        use `sketch.faces` property to access them."""

    @property
    def faces(self):
        """A 'flattened' grid of faces"""
        return self._faces

    @property
    def grid(self):
        """Use a single-tier grid by default; override the method for more sophistication."""
        return [self.faces]

    @property
    def center(self) -> NPPointType:
        """Center of this sketch"""
        return np.average([face.center for face in self.faces], axis=0)

    @property
    def positions(self) -> NPPointListType:
        """Reconstructs positions back from faces so they are always up-to-date,
        even after transforms"""
        indexes = list(np.array(self.indexes).flatten())
        max_index = max(indexes)
        all_points = f.flatten_2d_list([face.point_array.tolist() for face in self.faces])

        return np.array([all_points[indexes.index(i)] for i in range(max_index + 1)])
