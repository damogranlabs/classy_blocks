import warnings
from typing import Optional

import numpy as np

from classy_blocks.construct.array import Array
from classy_blocks.construct.curves.curve import PointCurveBase
from classy_blocks.types import NPPointListType, NPPointType, PointListType, PointType
from classy_blocks.util import functions as f


class DiscreteCurve(PointCurveBase):
    """A curve, defined by a set of points;
    All operations on this curve involve only the specified
    points with no interpolation (contrary to *InterpolatedCurves where
    values between points are interpolated).

    Parameter is actually an index to a given point;
    Discretization yields the original points;
    Length just sums the distances between points."""

    def __init__(self, points: PointListType):
        self.array = Array(points)
        self.bounds = (0, len(self.array) - 1)

    def discretize(
        self, param_from: Optional[float] = None, param_to: Optional[float] = None, _count: int = 0
    ) -> NPPointListType:
        """Discretizes this curve into points.

        With DiscreteCurve, parameter 'count' is ignored as points are taken directly."""
        param_from, param_to = self._get_params(param_from, param_to)
        param_start = int(min(param_from, param_to))
        param_end = int(max(param_from, param_to))

        discretized = self.array[param_start : param_end + 1]

        if param_from > param_to:
            return np.flip(discretized, axis=0)

        return discretized

    def get_length(self, param_from: Optional[float] = None, param_to: Optional[float] = None) -> float:
        """Returns the length of this curve between specified params."""
        return f.polyline_length(self.discretize(param_from, param_to))

    def get_closest_param(self, point: PointType) -> float:
        """Returns the index of point on this curve where distance to supplied
        point is the smallest.

        For DiscreteCurve, argument param_start is ignored since all points are checked."""
        return super().get_closest_param(point)

    @property
    def center(self):
        warnings.warn("Using an approximate default curve center (average)!", stacklevel=2)
        return np.average(self.discretize(), axis=0)

    def get_point(self, param: float) -> NPPointType:
        param = self._check_param(param)
        index = int(param)
        return self.array[index]

    @property
    def parts(self):
        return [self.array]
