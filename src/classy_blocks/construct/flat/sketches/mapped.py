from typing import List

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.types import IndexType, NPPointType, PointListType


class MappedSketch(Sketch):
    """A sketch that is created from predefined points.
    The points are connected to form quads which define Faces."""

    def __init__(self, positions: PointListType, quads: List[IndexType]):
        self._faces: List[Face] = []

        for quad in quads:
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

    # def smooth(self, n_iter: int = 5) -> None:
    #    """Smooth the internal points using laplacian smoothing"""
    #    for _ in range(n_iter):
    #        self.quad_map.smooth_laplacian()
