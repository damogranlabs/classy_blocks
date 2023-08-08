from typing import Callable, List, Optional

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.types import NPPointType, PointType
from classy_blocks.util.constants import DTYPE


class LineClamp(ClampBase):
    """Clamp that restricts point movement
    during optimization to a line, defined by 2 points;

    Parameter 't' goes from 0 at point_1 to 1 at point_2
    (and beyond if different bounds are specified)."""

    def __init__(self, vertex: Vertex, point_1: PointType, point_2: PointType, bounds: Optional[List[float]] = None):
        point_1 = np.array(point_1, dtype=DTYPE)
        point_2 = np.array(point_2, dtype=DTYPE)
        vector = point_2 - point_1

        if bounds is not None:
            clamp_bounds = [bounds]
        else:
            clamp_bounds = None

        super().__init__(vertex, lambda params: point_1 + params[0] * vector, clamp_bounds)

    @property
    def initial_params(self) -> List[float]:
        if self.bounds is None:
            return [0.5]

        return [(self.bounds[0][0] + self.bounds[0][1]) / 2.0]


class ParametricCurveClamp(ClampBase):
    """Clamp that restricts point movement during optimization
    to an analytically defined function p = f(t);

    Function f must take a single parameter 't' and return a point in 3D space."""

    def __init__(
        self, vertex: Vertex, function: Callable[[List[float]], NPPointType], bounds: Optional[List[float]] = None
    ):
        if bounds is not None:
            clamp_bounds = [bounds]
        else:
            clamp_bounds = None

        super().__init__(vertex, function, clamp_bounds)

    @property
    def initial_params(self):
        if self.bounds is None:
            return [0.5]

        return [(self.bounds[0][0] + self.bounds[0][1]) / 2.0]


# class InterpolatedCurveClamp(ClampBase):
#     """Clamp that restricts point movement during optimization
#     to a curve, interpolated between provided points"""


# class CircleClamp(ClampBase):
#     """Clamp that restricts point movement during optimization
#     to a circle, defined by a center, radius and normal"""
