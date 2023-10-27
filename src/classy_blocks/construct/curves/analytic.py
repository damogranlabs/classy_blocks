from typing import Tuple

import numpy as np

from classy_blocks.construct.curves.curve import FunctionCurveBase
from classy_blocks.construct.point import Point
from classy_blocks.types import NPVectorType, ParamCurveFuncType, PointType, VectorType
from classy_blocks.util import functions as f


class AnalyticCurve(FunctionCurveBase):
    """A parametric curve, defined by a user-specified function

    `P = f(t)`"""

    def __init__(self, function: ParamCurveFuncType, bounds: Tuple[float, float]):
        self.function = function
        self.bounds = bounds

    @property
    def parts(self):
        raise NotImplementedError("Transforming arbitrary analytic curves is currently not supported")


class LineCurve(AnalyticCurve):
    """A simple line, defined by 2 points.
    Parameter goes from 0 at point_1 to 1 at point_2.

    To extend the line  beyond given points, provide custom 'bounds'."""

    def __init__(self, point_1: PointType, point_2: PointType, bounds: Tuple[float, float] = (0, 1)):
        self.point_1 = Point(point_1)
        self.point_2 = Point(point_2)

        super().__init__(lambda t: self.point_1.position + self.vector * t, bounds)

    @property
    def vector(self) -> NPVectorType:
        return self.point_2.position - self.point_1.position

    @property
    def parts(self):
        return [self.point_1, self.point_2]

    @property
    def center(self):
        # this one is easy
        return (self.point_1.position + self.point_2.position) / 2


class CircleCurve(AnalyticCurve):
    """A parametric circle, defined by center, starting point and normal.
    A full circle is valid by default. Provide custom bounds to clip
    this curve to an arc."""

    def __init__(
        self, origin: PointType, rim: PointType, normal: VectorType, bounds: Tuple[float, float] = (0, 2 * np.pi)
    ):
        self.origin = Point(origin)
        self.rim = Point(rim)

        # normal is a unit vector and is not transformed the same
        # as points. To keep things simple, use (and transform) 3 points
        # and calculate normal on-the-go
        normal = f.unit_vector(normal)
        self.atop = Point(origin + normal)

        super().__init__(lambda t: f.rotate(self.rim.position, t, self.normal, self.origin.position), bounds)

    @property
    def normal(self) -> NPVectorType:
        return self.atop.position - self.origin.position

    @property
    def center(self):
        return self.origin.position

    @property
    def parts(self):
        return [self.origin, self.rim, self.atop]
