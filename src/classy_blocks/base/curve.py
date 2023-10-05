from typing import Callable, List, Optional

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize

from classy_blocks.types import NPPointListType, NPPointType, PointListType, PointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE, TOL


class Curve:
    """A parametric/parameterized curve in 3D space <point> = f(t)
    TODO: TEST"""

    eps = TOL / 10

    def __init__(self, function: Callable[[float], NPPointType], bounds: List[float]):
        self.function = function
        self.bounds = bounds

    def discretize(
        self, t_from: Optional[float] = None, t_to: Optional[float] = None, count: int = 15
    ) -> NPPointListType:
        """Returns discretized points by using evenly-spaced parameters.
        If from/to parameters are not given, respective bound is taken instead.

        TODO:
        - discretization based on deflection
        - with evenly-spaced output points
        - replace list comprehension with a vectorized operation"""
        if t_from is None:
            t_from = self.bounds[0]

        if t_to is None:
            t_to = self.bounds[1]

        return np.array([self.function(t) for t in np.linspace(t_from, t_to, num=count)])

    def get_closest_param(self, position: PointType) -> float:
        """Finds the param on curve where point is the closest to given"""
        position = np.array(position, dtype=DTYPE)

        return scipy.optimize.minimize_scalar(lambda t: f.norm(self.function(t) - position), bounds=self.bounds).x

    def get_length(self, t_from: float, t_to: float) -> float:
        """Length of the curve within given bounds"""

        # def dr_dt_mag(t):
        #    return f.norm(self.function(t + self.eps / 2) - self.function(t - self.eps / 2)) / self.eps

        # return scipy.integrate.quad(dr_dt_mag, t_from, t_to, epsabs=self.eps)[0]

        # TODO: check for discontinuities and use the above if there are none
        d_this = self.discretize(t_from, t_to, count=100)
        d_next = np.roll(d_this, 1)

        norms = np.linalg.norm(d_this[:-1] - d_next[1:], axis=0)

        return np.sum(norms)

    @property
    def length(self) -> float:
        """Returns full length of the curve between provided bounds"""
        return self.get_length(self.bounds[0], self.bounds[1])

    @classmethod
    def from_points(cls, points: PointListType, t_from: float, t_to: float) -> "Curve":
        """Returns an interpolation function point = f(t)"""
        points = np.array(points, dtype=DTYPE)

        count = len(points)
        ts = np.linspace(t_from, t_to, num=count)
        xs = points.T[0]
        ys = points.T[1]
        zs = points.T[2]

        fx = scipy.interpolate.interp1d(ts, xs, bounds_error=False, fill_value="extrapolate")
        fy = scipy.interpolate.interp1d(ts, ys, bounds_error=False, fill_value="extrapolate")
        fz = scipy.interpolate.interp1d(ts, zs, bounds_error=False, fill_value="extrapolate")

        def function(t: float):
            return np.array([fx(t), fy(t), fz(t)])

        return cls(function, [t_from, t_to])
