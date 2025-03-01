import abc

import numpy as np
import scipy.interpolate

from classy_blocks.cbtyping import FloatListType, NPPointType, ParamCurveFuncType
from classy_blocks.construct.series import Series


class InterpolatorBase(abc.ABC):
    """A easy wrapper around all the complicated options
    of bunches of scipy's interpolation routines.

    Also provides caching of built functions unless a transformation
    has been made (a.k.a. invalidate()) has been called."""

    @abc.abstractmethod
    def _get_function(self) -> ParamCurveFuncType:
        """Returns an interpolation function from stored points"""

    def __init__(self, points: Series, extrapolate: bool, equalize: bool = True):
        self.points = points
        self.extrapolate = extrapolate
        self.equalize = equalize

        self.function = self._get_function()
        self._valid = True

    def __call__(self, param: float) -> NPPointType:
        if not self._valid:
            self.function = self._get_function()
            self._valid = True

        return self.function(param)

    def invalidate(self) -> None:
        self._valid = False

    @property
    def params(self) -> FloatListType:
        """A list of parameters for the interpolation curve.
        If not equalized, it's just linearly spaced floats;
        if equalized, scaled distances between provided points are taken so that
        evenly spaced parameters will produce evenly spaced points even if
        interpolation points are unequally spaced."""
        if self.equalize:
            lengths = np.cumsum(np.sqrt(np.sum((self.points[:-1] - self.points[1:]) ** 2, axis=1)))
            return np.concatenate(([0], lengths / lengths[-1]))

        return np.linspace(0, 1, num=len(self.points))


class LinearInterpolator(InterpolatorBase):
    def _get_function(self):
        if self.extrapolate:
            bounds_error = False
            fill_value = "extrapolate"
        else:
            bounds_error = True
            fill_value = np.nan

        function = scipy.interpolate.interp1d(
            self.params,
            self.points.points,
            bounds_error=bounds_error,
            fill_value=fill_value,  # type: ignore
            axis=0,  # type: ignore
        )

        return lambda param: function(param)


class SplineInterpolator(InterpolatorBase):
    def _get_function(self):
        spline = scipy.interpolate.make_interp_spline(self.params, self.points.points, check_finite=False)

        return lambda t: spline(t, extrapolate=self.extrapolate)
