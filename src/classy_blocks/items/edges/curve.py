import dataclasses

from classy_blocks.construct import edges
from classy_blocks.items.edges.edge import Edge
from classy_blocks.types import NPPointListType
from classy_blocks.util.constants import vector_format


@dataclasses.dataclass
class CurveEdge(Edge):
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
        return self.data.curve.discretize(self.param_start, self.param_end)[1:-1]

    @property
    def description(self):
        # this is a curve edge but for blockMesh, it's a spline
        edge_def = super().description
        edge_def = edge_def.replace("curve", "spline")

        point_list = " ".join([vector_format(p) for p in self.point_array])
        return edge_def + "(" + point_list + ")"
