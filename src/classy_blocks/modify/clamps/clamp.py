import abc
from typing import Callable, List, Optional

import scipy.optimize

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class ClampBase(abc.ABC):
    """Movement restriction for optimization by vertex movement"""

    def __init__(
        self,
        vertex: Vertex,
        position_function: Callable[[List[float]], NPPointType],
        bounds: Optional[List[List[float]]] = None,
    ):
        self.vertex = vertex
        self.position_function = position_function
        self.bounds = bounds

        self.params = self.get_params_from_vertex()
        self.update_params(self.params)

    @property
    @abc.abstractmethod
    def initial_params(self) -> List[float]:
        """Returns initial guess for get_params_from_vertex"""

    def get_params_from_vertex(self) -> List[float]:
        """Returns parameters from initial vertex position"""

        def distance_from_vertex(params):
            return f.norm(self.point - self.position_function(params))

        result = scipy.optimize.minimize(
            distance_from_vertex, self.initial_params, bounds=self.bounds, method="SLSQP", tol=TOL
        )
        return result.x

    def update_params(self, params: List[float]):
        self.params = params
        self.vertex.move_to(self.position_function(self.params))

    @property
    def point(self) -> NPPointType:
        """Point according to current parameters"""
        return self.vertex.position
