import dataclasses

import numpy as np

from classy_blocks.construct import edges
from classy_blocks.items.edges.edge import Edge
from classy_blocks.util import constants

from classy_blocks.util import functions as f


@dataclasses.dataclass
class SplineEdge(Edge):
    """Spline edge, defined by multiple points"""

    data: edges.Spline

    @property
    def length(self):
        # just sum distances between defining points
        all_points = np.concatenate(([self.vertex_1.pos], self.data.through, [self.vertex_2.pos]))
        shifted = np.roll(all_points, 1, axis=0)

        distances = (all_points - shifted)[1:]

        return np.sum([f.norm(d) for d in distances])

    @property
    def description(self):
        point_list = " ".join([constants.vector_format(p) for p in self.data.through])
        return super().description + "(" + point_list + ")"


@dataclasses.dataclass
class PolyLineEdge(SplineEdge):
    """PolyLine variant of SplineEdge"""

    data: edges.PolyLine

    @property
    def length(self):
        # same as spline but more accurate :)
        return super().length
