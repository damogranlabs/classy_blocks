import dataclasses

import numpy as np

from classy_blocks.data import edges
from classy_blocks.items.edges.edge import Edge
from classy_blocks.util import constants
from classy_blocks.util import functions as f

@dataclasses.dataclass
class SplineEdge(Edge):
    """Spline edge, defined by multiple points"""
    data: edges.Spline

    @property
    def length(self):
        # just sum distances between self.points
        points = np.asarray([point for point in self.data.points])
        shifted = np.roll(points, 1, axis=0)

        return np.sum(f.norm(points - shifted)[1:])

    @property
    def description(self):
        point_list = ' '.join([constants.vector_format(p) for p in self.data.points])
        return super().description + '(' + point_list + ')'

@dataclasses.dataclass
class PolyLineEdge(SplineEdge):
    """PolyLine variant of SplineEdge"""
    data: edges.PolyLine

    @property
    def length(self):
        # same as spline but more accurate :)
        return super().length
