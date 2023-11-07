import abc
import warnings
from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.point import Point
from classy_blocks.types import NPPointListType, NPPointType, ParamCurveFuncType, PointListType, PointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE, TOL


class CurveBase(ElementBase):
    """A parametric/analytic/interpolated curve in 3D space: <point> = f(t)"""

    bounds: Tuple[float, float]

    def _check_param(self, param: float) -> int:
        """Checks that the passed parameter is legit for the given set of points"""
        param = int(param)

        if not self.bounds[0] <= param <= self.bounds[1]:
            raise ValueError(f"Invalid parameter {param} (0...{self.bounds[1]})")

        return param

    def _get_params(self, param_from: Optional[float] = None, param_to: Optional[float] = None) -> Tuple[float, float]:
        """Always take lower/upper bound if params are not supplied"""
        if param_from is None:
            param_from = self.bounds[0]
        if param_to is None:
            param_to = self.bounds[1]

        self._check_param(param_from)
        self._check_param(param_to)

        if param_from == param_to:
            raise ValueError("Provide two different parameters for discretization (or use get_point())")

        return param_from, param_to

    @abc.abstractmethod
    def get_point(self, param: float) -> NPPointType:
        """Returns point at given parameter"""

    @abc.abstractmethod
    def discretize(
        self, param_from: Optional[float] = None, param_to: Optional[float] = None, count: int = 10
    ) -> NPPointListType:
        """Discretizes this curve into 'count' points.
        Optionally, use the curve between passed parameters; default 'count' is chosen
        as a sane default for a blockMesh edge."""

    @abc.abstractmethod
    def get_length(self, param_from: Optional[float] = None, param_to: Optional[float] = None) -> float:
        """Returns the length of the curve between the given parameters;
        bounds are used if they are not supplied."""

    @property
    def length(self) -> float:
        """Returns full length of the curve between provided bounds"""
        return self.get_length(self.bounds[0], self.bounds[1])

    @abc.abstractmethod
    def get_closest_param(self, point: PointType) -> float:
        """Finds the parameter on curve where point is the closest to given point;
        To improve search speed and reliability, an optional starting
        estimation can be supplied."""
        # because curves can have all sorts of shapes, find
        # initial guess by checking distance to discretized points
        point = np.array(point)
        all_points = self.discretize()

        distances = np.array([f.norm(p - point) for p in all_points])
        params = np.linspace(self.bounds[0], self.bounds[1], num=len(distances))

        i_distance = np.argmin(distances)
        return params[i_distance]


class PointCurveBase(CurveBase):
    """A base object for curves, defined by a list of points"""

    @staticmethod
    def _check_points(points: PointListType) -> List[Point]:
        """Check that provided points are sufficient for a curve"""
        points = np.array(points, dtype=DTYPE)
        shape = np.shape(points)

        if shape[0] < 2:
            raise ValueError("Provide at least 2 points that represent a curve")

        if shape[1] != 3:
            raise ValueError("Provide points in 3D space")

        return [Point(p) for p in points]

    @property
    def center(self):
        warnings.warn("Using an approximate default curve center (average)!", stacklevel=2)
        return np.average(self.discretize(), axis=0)


class FunctionCurveBase(PointCurveBase):
    """A base object for curves, driven by functions"""

    function: ParamCurveFuncType

    def discretize(
        self, param_from: Optional[float] = None, param_to: Optional[float] = None, count: int = 15
    ) -> NPPointListType:
        """Discretized the curve into 'count' points."""
        param_from, param_to = self._get_params(param_from, param_to)
        params = np.linspace(param_from, param_to, num=count)

        return np.array([self.function(t) for t in params])

    def get_closest_param(self, point: PointType) -> float:
        """Finds the param on curve where point is the closest to given point;
        To improve search speed and reliability, an optional starting
        estimation can be supplied."""
        param_start = super().get_closest_param(point)
        point = np.array(point)

        result = scipy.optimize.minimize(
            lambda t: f.norm(self.get_point(t[0]) - point), (param_start,), bounds=(self.bounds,)
        )

        return result.x[0]

    def get_point(self, param: float) -> NPPointType:
        self._check_param(param)
        return self.function(param)

    def get_length(self, param_from: Optional[float] = None, param_to: Optional[float] = None) -> float:
        """Returns the length of this curve by numerical
        integration of segments."""

        def dr_dt_mag(t):
            return f.norm(self.get_point(t + TOL / 2) - self.get_point(t - TOL / 2)) / TOL

        return scipy.integrate.quad(dr_dt_mag, param_from, param_to, epsabs=TOL)[0]
