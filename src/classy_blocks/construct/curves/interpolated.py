import abc
from typing import Optional, Type

import numpy as np

from classy_blocks.construct.curves.curve import FunctionCurveBase
from classy_blocks.construct.curves.interpolators import InterpolatorBase, LinearInterpolator, SplineInterpolator
from classy_blocks.types import PointListType


class InterpolatedCurveBase(FunctionCurveBase, abc.ABC):
    """A curve, obtained by interpolation between provided points;
    Unlike DiscreteCurve, all values between points are accessible by
    providing appropriate parameter.

    The parameter is similar to DiscreteCurve's, like an index to
    the nearest point but here all non-integer values in between
    are available too.

    An interpolation function is build from provided points.
    Length, discretization, center and other calculated properties
    are based on that function rather than specified points."""

    _interpolator: Type[InterpolatorBase]

    def __init__(self, points: PointListType):
        self.points = self._check_points(points)
        self.function = self._interpolator(self.points, False)
        self.bounds = (0, 1)

    @property
    def segments(self) -> int:
        """Returns number of points this curve was created from"""
        return len(self.points) - 1

    @property
    def parts(self):
        # This is called when a transform of any kind is requested on
        # this class; that means the interpolation function
        # is no longer valid and needs to be rebuilt
        self.function.invalidate()

        return self.points


class LinearInterpolatedCurve(InterpolatedCurveBase):
    _interpolator = LinearInterpolator

    def get_length(self, param_from: Optional[float] = None, param_to: Optional[float] = None) -> float:
        """Returns the length of this curve by summing distance between
        points. The 'count' parameter is ignored as the original points are taken."""
        # TODO: use the same function as items.edges.spline
        param_from, param_to = self._get_params(param_from, param_to)

        index_from = int(param_from * self.segments) + 1
        index_to = int(param_to * self.segments)

        if index_from < index_to:
            indexes = list(range(index_from, index_to + 1))
        else:
            indexes = []

        params = [param_from, *[i / self.segments for i in indexes], param_to]
        discretized = np.array([self.function(t) for t in params])
        return np.sum(np.sqrt(np.sum((discretized[:-1] - discretized[1:]) ** 2, axis=1)))


class SplineInterpolatedCurve(InterpolatedCurveBase):
    _interpolator = SplineInterpolator
