from typing import Tuple

from classy_blocks.construct.curves.curve import FunctionCurveBase
from classy_blocks.construct.point import Point
from classy_blocks.types import NPVectorType, ParamCurveFuncType, PointType


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


# class CircleCurve(AnalyticCurve):
# TODO!
