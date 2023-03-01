from typing import List

from classy_blocks.items.vertex import Vertex

class VertexList:
    """ Handling of the 'vertices' part of blockMeshDict """

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
    def vertices(self) -> List[Vertex]:
        """A convenient access to Vertex.registry"""
        return Vertex.registry

    @property
    def description(self) -> str:
        """Output for blockMeshDict"""
        out = 'vertices\n(\n'

        for vertex in Vertex.registry:
            out += f"\t{vertex.description}\n"

        out += ");\n\n"

        return out
