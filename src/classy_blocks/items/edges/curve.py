import abc
import dataclasses

import numpy as np

from classy_blocks.construct import edges
from classy_blocks.construct.curves.discrete import DiscreteCurve
from classy_blocks.items.edges.edge import Edge
from classy_blocks.types import EdgeKindType, NPPointListType
from classy_blocks.util.constants import vector_format


class CurveEdgeBase(Edge, abc.ABC):
    """Base class for edges of any curved shape,
    defined as a list of points"""

    data: edges.OnCurve

    @property
    @abc.abstractmethod
    def point_array(self) -> NPPointListType:
        """Edge points as numpy array"""

    @property
    def param_start(self) -> float:
        return 0

    @property
    def param_end(self) -> float:
        return self.data.n_points - 1

    @property
    def description(self):
        point_list = " ".join([vector_format(p) for p in self.point_array])
        return super().description + "(" + point_list + ")"


@dataclasses.dataclass
class SplineEdge(CurveEdgeBase):
    """Spline edge, defined by multiple points"""

    data: edges.Spline

    @property
    def point_array(self) -> NPPointListType:
        return self.data.discretize(self.param_start, self.param_end)

    @property
    def length(self):
        points = np.concatenate(([self.vertex_1.position], self.point_array, [self.vertex_2.position]))

        return DiscreteCurve(points).length


@dataclasses.dataclass
class PolyLineEdge(SplineEdge):
    """PolyLine variant of SplineEdge"""

    data: edges.PolyLine


@dataclasses.dataclass
class OnCurveEdge(CurveEdgeBase):
    """Spline edge, defined by a parametric curve"""

    data: edges.OnCurve

    @property
    def length(self):
        return self.data.curve.get_length(self.param_start, self.param_end)

    @property
    def param_start(self) -> float:
        """Parameter of given curve at vertex 1"""
        return self.data.curve.get_closest_param(self.vertex_1.position)

    @property
    def param_end(self) -> float:
        """Parameter of given curve at vertex 2"""
        return self.data.curve.get_closest_param(self.vertex_2.position)

    @property
    def point_array(self) -> NPPointListType:
        return self.data.discretize(self.param_start, self.param_end)[1:-1]

    @property
    def representation(self) -> EdgeKindType:
        return self.data.representation
