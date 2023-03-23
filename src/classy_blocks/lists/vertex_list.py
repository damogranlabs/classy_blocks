from typing import List, Dict, Optional

import numpy as np

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.types import NPPointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from classy_blocks.items.vertex import Vertex

class VertexList:
    """ Handling of the 'vertices' part of blockMeshDict """
    def __init__(self):
        self.vertices:List[Vertex] = []

        # a collection of duplicated vertices
        # belonging to a certain patch name
        self.duplicated:Dict[str, List[Vertex]] = {}

    def find(self, position:NPPointType) -> Vertex:
        """checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex"""
        # TODO: optimize (octree/kdtree from scipy) (?)
        for vertex in self.vertices:
            if f.norm(vertex.pos - position) < constants.TOL:
                return vertex

        raise VertexNotFoundError(f"Vertex not found: {str(position)}")

    def find_duplicated(self, position:NPPointType, slave_patch:str) -> Vertex:
        """Finds an already duplicated vertex on a slave patch; raises
        a VertexNotFoundError if there's no such vertex yet"""
        for vertex in self.duplicated.get(slave_patch, []):
            if f.norm(vertex.pos - position) < constants.TOL:
                return vertex
        
        raise VertexNotFoundError(f"No duplicated vertex found at {str(position)}")

    def add(self, point:NPPointType, slave_patch:Optional[str]=None) -> Vertex:
        """Re-use existing vertices when there's already one at the position;
        unless that vertex belongs to a slave of a face-merged pair - 
        in that case add a duplicate in the same position anyway"""
        # TODO: TEST
        # TODO: prettify
        if slave_patch is not None:
            try:
                return self.find_duplicated(point, slave_patch)
            except VertexNotFoundError:
                vertex = Vertex(point, len(self.vertices))
                self.vertices.append(vertex)

                if slave_patch is not None:
                    if slave_patch not in self.duplicated:
                        self.duplicated[slave_patch] = []

                    self.duplicated[slave_patch].append(vertex)
                
                return vertex

        try:
            vertex = self.find(point)
        except VertexNotFoundError:
            # no vertex was found, add a new one;
            vertex = Vertex(point, len(self.vertices))
            self.vertices.append(vertex)

            if slave_patch is not None:
                if slave_patch not in self.duplicated:
                    self.duplicated[slave_patch] = []

                self.duplicated[slave_patch].append(vertex)

        return vertex

    @property
    def description(self) -> str:
        """Output for blockMeshDict"""
        out = 'vertices\n(\n'

        for vertex in self.vertices:
            out += f"\t{vertex.description}\n"

        out += ");\n\n"

        return out