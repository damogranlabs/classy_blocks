import dataclasses

from classy_blocks.construct import edges
from classy_blocks.items.edges.spline import SplineEdge
from classy_blocks.types import EdgeKindType, NPPointListType


@dataclasses.dataclass
class OnCurveEdge(SplineEdge):
    """Spline edge, defined by a parametric curve"""

    # TODO: TEST

    data: edges.OnCurve

    @property
    def param_start(self) -> float:
        """Parameter of given curve at vertex 1"""
        return self.data.curve.get_closest_param(self.vertex_1.position)

    @property
    def param_end(self) -> float:
        """Parameter of given curve at vertex 2"""
        return self.data.curve.get_closest_param(self.vertex_2.position)

    @property
    def length(self):
        return self.data.curve.get_length(self.param_start, self.param_end)

    @property
    def point_array(self) -> NPPointListType:
        """spline points as numpy array"""
        return self.data.curve.discretize(self.param_start, self.param_end, count=self.data.n_points + 2)[1:-1]

    @property
    def representation(self) -> EdgeKindType:
        return self.data.representation
