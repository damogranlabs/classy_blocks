import abc
from typing import Callable, List, Optional

import numpy as np
import scipy.optimize

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.links import LinkBase
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
        initial_params: Optional[List[float]] = None,
    ):
        self.vertex = vertex
        self.position_function = position_function
        self.bounds = bounds
        self.initial_params = initial_params
        self.links: List[LinkBase] = []

        self.params = self.get_params_from_vertex()
        self.update_params(self.params)

    @property
    @abc.abstractmethod
    def initial_guess(self) -> List[float]:
        """Returns initial guess for get_params_from_vertex"""

    def get_params_from_vertex(self) -> List[float]:
        """Returns parameters from initial vertex position"""

        def distance_from_vertex(params):
            return f.norm(self.point - self.position_function(params))

        result = scipy.optimize.minimize(distance_from_vertex, self.initial_guess, bounds=self.bounds, tol=TOL)

        return result.x

    def update_params(self, params: List[float], relaxation: float = 1):
        """Updates parameters to given. Optionally limit the change
        using the same logic as under-relaxation in CFD."""
        old_params = np.array(self.params)
        new_params = np.array(params)

        relaxed = old_params + relaxation * (new_params - old_params)

        self.params = relaxed.tolist()
        self.vertex.move_to(self.position_function(self.params))

        self._update_links()

    def _update_links(self) -> None:
        for link in self.links:
            link.update()

    @property
    def point(self) -> NPPointType:
        """Point according to current parameters"""
        return self.vertex.position

    def add_link(self, link: LinkBase) -> None:
        # TODO: check for duplicates?
        self.links.append(link)

    @property
    def is_linked(self) -> bool:
        return len(self.links) > 0
