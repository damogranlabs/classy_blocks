from typing import Optional, Set

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.find.finder import FinderBase
from classy_blocks.types import PointType


class GeometricFinder(FinderBase):
    """Find mesh vertices inside a specified geometric shape"""

    def find_in_sphere(self, position: PointType, radius: Optional[float] = None) -> Set[Vertex]:
        """Returns a list of vertices that are
        inside a sphere of given radius; if that is not given,
        constants.TOL is taken"""
        return self._find_by_position(position, radius)

    def find_in_box_corners(self, corner_point: PointType, diagonal_point: PointType) -> Set[Vertex]:
        """Returns a list of vertices that are inside a box, defined
        by two points on each end of volumetric diagonal."""
        # TODO: un-wip this
        raise NotImplementedError("Alas, this is a work-in-progress")

    def find_in_box_center(self, center_point: PointType, size_x: float, size_y: float, size_z: float) -> Set[Vertex]:
        """Returns a list of vertices that are inside a box,
        defined by its center and width, height and depth."""
        # TODO: un-wip this
        raise NotImplementedError("Alas, this is a work-in-progress" "")
