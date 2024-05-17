from typing import List, Tuple

import numpy as np

from classy_blocks.construct.flat.quad import Quad, smooth
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.types import NPPointType, PointListType
from classy_blocks.util.constants import DTYPE


class MappedSketch(Sketch):
    """A sketch that is created from predefined points.
    The points are connected to form quads which define Faces."""

    def __init__(self, positions: PointListType, quads: List[Tuple[int, int, int, int]], smooth_iter: int = 0):
        positions = np.array(positions, dtype=DTYPE)

        if smooth_iter > 0:
            positions = smooth(positions, quads, smooth_iter)

        self.quads = [Quad(positions, quad) for quad in quads]

        self.add_edges()

    def add_edges(self) -> None:
        """An optional method that will add edges to wherever they need to be."""

    @property
    def faces(self):
        """A 'flattened' grid"""
        return [quad.face for quad in self.quads]

    @property
    def grid(self):
        """Use a single-tier grid by default; override the method for more sophistication."""
        return [self.faces]

    @property
    def center(self) -> NPPointType:
        """Center of this sketch"""
        return np.average([face.center for face in self.faces], axis=0)
