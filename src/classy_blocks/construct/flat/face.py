from typing import List

import numpy as np

from classy_blocks.types import PointListType, VectorType
from classy_blocks.data.data_object_base import DataObjectBase

from classy_blocks.data.edge_data import EdgeData

from classy_blocks.util import constants
from classy_blocks.util import functions as f

class Face(DataObjectBase):
    """A collection of 4 Vertices and optionally 4 Edges,
    creating an arbitrary quadrangle. A Block can be made from 2 faces,
    connected with straight lines or, again, (optionally) curved edges."""
    def __init__(self, points: PointListType, check_coplanar: bool = False):
        super().__init__(points)

        if np.shape(self.points) != (4, 3):
            raise ValueError("Provide exactly 4 points in 3D space")
        
        if check_coplanar:
            pnt = self.points
            if abs(
                np.dot(
                    pnt[1] - pnt[0],
                    np.cross(pnt[3] - pnt[0], pnt[2] - pnt[0])
                )) > constants.tol:
                raise ValueError("Points are not coplanar!")

            # TODO: coplanar edges?

    @property
    def top_edges(self) -> List[EdgeData]:
        """Returns a list of edges, defined between top corners of a block"""
        return []
    
    @property
    def bottom_edges(self) -> List[EdgeData]:
        """Returns a list of edges, defined between bottom corners of a block"""
        return []

    def invert(self):
        """Reverses the order of points in this face."""
        self.points = np.flip(self.points, axis=0)

    @property
    def normal(self) -> VectorType:
        """Returns a vector normal to this face.
        For non-planar faces the same rule as in OpenFOAM is followed:
        divide a quadrangle into 4 triangles, each joining at face center;
        a normal is the average of normals of those triangles."""
        # TODO: TEST
        # TODO: cache?
        points = self.points
        center = self.center

        side_1 = points - center
        side_2 = np.roll(points, -1, axis=0) - center
        normals = np.cross(side_1, side_2)

        return np.average(normals, axis=0)
