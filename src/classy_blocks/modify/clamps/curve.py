import copy
from typing import List, Optional, Tuple

from classy_blocks.construct.curves.analytic import LineCurve
from classy_blocks.construct.curves.curve import CurveBase
from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f


class CurveClamp(ClampBase):
    """Clamp that restricts point movement during optimization
    to a predefined curve.

    The curve parameter that corresponds to given vertex's position
    is obtained automatically by minimization. To provide a better starting
    point in case minimization fails or produces wrong results,
    an initial parameter can be supplied."""

    def __init__(
        self,
        vertex: Vertex,
        curve: CurveBase,
        initial_param: Optional[float] = None,
    ):
        if initial_param is not None:
            initial = [initial_param]
        else:
            initial = [curve.get_closest_param(vertex.position)]

        super().__init__(vertex, lambda t: curve.get_point(t[0]), [list(curve.bounds)], initial)

    @property
    def initial_guess(self):
        return self.initial_params


class LineClamp(ClampBase):
    """Clamp that restricts point movement
    during optimization to a line, defined by 2 points;

    Parameter 't' goes from 0 at point_1 to 1 at point_2
    (and beyond if different bounds are specified)."""

    def __init__(self, vertex: Vertex, point_1: PointType, point_2: PointType, bounds: Tuple[float, float] = (0, 1)):
        curve = LineCurve(point_1, point_2, bounds)

        super().__init__(vertex, lambda t: curve.get_point(t[0]), [list(bounds)])

    @property
    def initial_guess(self) -> List[float]:
        # Finding the closest point on a line is reliable enough
        # so that specific initial parameters are not needed
        return [0]


class RadialClamp(ClampBase):
    """Clamp that restricts point movement during optimization
    to a circular trajectory, defined by center, normal and
    vertex position at clamp initialization.

    Parameter t goes from 0 at initial vertex position to 2*pi
    at the same position all the way around the circle"""

    def __init__(self, vertex: Vertex, center: PointType, normal: VectorType, bounds: Optional[List[float]] = None):
        initial_point = copy.copy(vertex.position)

        if bounds is not None:
            clamp_bounds = [bounds]
        else:
            clamp_bounds = None

        super().__init__(vertex, lambda params: f.rotate(initial_point, params[0], normal, center), clamp_bounds)

    @property
    def initial_guess(self):
        return [0]
