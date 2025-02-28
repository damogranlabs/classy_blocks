import abc
from typing import Callable, List, Optional

import numpy as np
import scipy.optimize

from classy_blocks.cbtyping import NPPointType, PointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class ClampBase(abc.ABC):
    """Movement restriction for optimization by vertex movement"""

    def __init__(
        self,
        position: PointType,
        function: Callable[[List[float]], NPPointType],
        bounds: Optional[List[List[float]]] = None,
        initial_params: Optional[List[float]] = None,
    ):
        self.position = np.array(position)
        self.function = function
        self.bounds = bounds
        self.initial_params = initial_params

        self.params = self.get_params()

    @property
    @abc.abstractmethod
    def initial_guess(self) -> List[float]:
        """Returns initial guess for get_params_from_vertex"""

    def get_params(self) -> List[float]:
        """Returns parameters from initial vertex position"""

        def distance_from_vertex(params):
            return f.norm(self.position - self.function(params))

        result = scipy.optimize.minimize(distance_from_vertex, self.initial_guess, bounds=self.bounds, tol=TOL)

        return result.x

    def update_params(self, params: List[float]):
        """Updates parameters to given."""
        self.params = params
        self.position = self.function(self.params)
