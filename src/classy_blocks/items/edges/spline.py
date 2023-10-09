import dataclasses

import numpy as np

from classy_blocks.construct import edges
from classy_blocks.items.edges.edge import Edge
from classy_blocks.types import NPPointListType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import vector_format


@dataclasses.dataclass
class SplineEdge(Edge):
    """Spline edge, defined by multiple points"""

    data: edges.Spline

    @property
    def length(self):
        # just sum distances between defining points
        all_points = np.concatenate(([self.vertex_1.position], self.point_array, [self.vertex_2.position]))
        shifted = np.roll(all_points, 1, axis=0)

        distances = (all_points - shifted)[1:]

        return np.sum([f.norm(d) for d in distances])

    @property
    def point_array(self) -> NPPointListType:
        """spline points as numpy array"""
        return np.array([p.position for p in self.data.points])

    @property
    def description(self):
        point_list = " ".join([vector_format(p) for p in self.point_array])
        return super().description + "(" + point_list + ")"


@dataclasses.dataclass
class PolyLineEdge(SplineEdge):
    """PolyLine variant of SplineEdge"""

    data: edges.PolyLine

    @property
    def length(self):
        # same as spline but more accurate :)
        return super().length
