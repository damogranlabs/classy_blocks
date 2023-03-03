from typing import List

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

    def find(self, position:NPPointType) -> Vertex:
        """checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex"""
        # TODO: optimize (octree/kdtree from scipy) (?)
        for vertex in self.vertices:
            if f.norm(vertex.pos - np.asarray(position)) < constants.TOL:
                return vertex

        raise VertexNotFoundError(f"Vertex not found: {str(position)}")

    def add(self, point:NPPointType) -> Vertex:
        """Re-use existing vertices when there's already one at the position;"""
        try:
            vertex = self.find(point)
            # TODO: check for face-merged stuff
        except VertexNotFoundError:
            # no vertex was found, add a new one;
            vertex = Vertex(point, len(self.vertices))
            self.vertices.append(vertex)

        return vertex

        # merged patches: duplicate all points that define slave patches
        # duplicated_points = {}  # { original_index:new_vertex }
        # slave_patches = [mp[1] for mp in merged_patches]

        # for patch in slave_patches:
        #     for block in blocks:
        #         if patch in block.patches:
        #             patch_sides = block.get_patch_sides(patch)
        #             for side in patch_sides:
        #                 face_indexes = block.get_side_indexes(side, local=True)

        #                 for i in face_indexes:
        #                     vertex = block.vertices[i]

        #                     if vertex.mesh_index not in duplicated_points:
        #                         new_vertex = Vertex(vertex.point)
        #                         new_vertex.mesh_index = len(self.vertices)
        #                         self.vertices.append(new_vertex)

        #                         block.vertices[i] = new_vertex

        #                         duplicated_points[vertex.mesh_index] = new_vertex
        #                     else:
        #                         block.vertices[i] = duplicated_points[vertex.mesh_index]

    @property
    def description(self) -> str:
        """Output for blockMeshDict"""
        out = 'vertices\n(\n'

        for vertex in self.vertices:
            out += f"\t{vertex.description}\n"

        out += ");\n\n"

        return out