from typing import Tuple

from classy_blocks.construct.curves.curve import FunctionCurveBase
from classy_blocks.types import ParamCurveFuncType


class AnalyticCurve(FunctionCurveBase):
    """A parametric curve, defined by a user-specified function

    `P = f(t)`"""

    def __init__(self, function: ParamCurveFuncType, bounds: Tuple[float, float]):
        self.function = function
        self.bounds = bounds

    @property
    def parts(self):
        raise NotImplementedError("Transforming analytic curves is currently not supported")


# class CircleCurve(AnalyticCurve):
# TODO!

# class LineCurve(AnalyticCurve):
# TODO!
