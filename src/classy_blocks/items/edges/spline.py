import dataclasses

from typing import Callable, ClassVar, List

import numpy as np

from classy_blocks.items.edges.edge import Edge
from classy_blocks.types import PointType, PointListType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

@dataclasses.dataclass
class SplineEdge(Edge):
    """Spline edge, defined by multiple points"""

    kind: ClassVar[str] = "spline"

    points: PointListType

    def transform(self, function: Callable):
        self.points = np.array([function(p) for p in self.points])

        return self
    
    @property
    def args(self) -> List:
        return super().args + [self.points]

    @property
    def length(self):
        # just sum distances between self.points
        points = np.asarray(self.points)
        shifted = np.roll(points, 1, axis=0)

        return np.sum(f.norm(points - shifted)[1:])

    @property
    def description(self):
        point_list = ' '.join([constants.vector_format(p) for p in self.points])
        return super().description + '(' + point_list + ')'

@dataclasses.dataclass
class PolyLineEdge(SplineEdge):
    """PolyLine variant of SplineEdge"""

    kind: ClassVar[str] = "polyLine"

    @property
    def length(self):
        # same as spline but more accurate :)
        return super().length

