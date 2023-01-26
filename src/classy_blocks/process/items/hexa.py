import dataclasses

from typing import List

from classy_blocks.define import edge
from classy_blocks.define.block import Block
from classy_blocks.define import point

class Hexa:
    """Actual entry in blockMeshDict.blocks, generated from Block"""
    def __init__(self, block:Block, index:int):
        self.block = block
        self.index = index # position of this hex in the list

        self.vertices:List[point.Point] = []

    ###
    ### Information
    ###
    @property
    def points(self):
        return self.block.points

    # def get_side_vertices(self, orient:str) -> List[Vertex]:
    #     """Returns Vertices that define the given side"""
    #     return [self.vertices[i] for i in self.get_side_indexes(orient, local=True)]

    # def get_side_indexes(self, orient:str, local:bool=False) -> List[int]:
    #     """Returns block-local indexes of vertices that define this side"""
    #     if local:
    #         return c.FACE_MAP[orient]

    #     side_vertices = [self.vertices[i] for i in c.FACE_MAP[orient]]
    #     return [v.mesh_index for v in side_vertices]

    # def get_patch_sides(self, patch: str) -> List[str]:
    #     """Returns sides in this block that belong to a given patch"""
    #     if patch not in self.patches:
    #         return []

    #     orients = []

    #     for orient in c.FACE_MAP:
    #         if self.sides[orient].patch == patch:
    #             orients.append(orient)

    #     return orients



    # def find_edge(self, index_1: int, index_2: int) -> Edge:
    #     """Returns edges between given vertex indexes;
    #     the indexes in parameters refer to internal block numbering"""
    #     for e in self.edges:
    #         if {e.index_1, e.index_2} == {index_1, index_2}:
    #             return e

    #     return None

    # def get_size(self, axis: int, take: Literal["min", "max", "avg"] = "avg") -> float:
    #     """Returns block dimensions in given axis"""
    #     # if an edge is defined, use the edge.get_length(),
    #     # otherwise simply distance between two points
    #     def vertex_distance(index_1, index_2):
    #         return f.norm(self.vertices[index_1].point - self.vertices[index_2].point)

    #     def block_size(index_1, index_2):
    #         edge = self.find_edge(index_1, index_2)
    #         if edge:
    #             return edge.get_length()

    #         return vertex_distance(index_1, index_2)

    #     edge_lengths = [block_size(pair[0], pair[1]) for pair in c.AXIS_PAIRS[axis]]

    #     if take == "avg":
    #         return sum(edge_lengths) / len(edge_lengths)

    #     if take == "min":
    #         return min(edge_lengths)

    #     if take == "max":
    #         return max(edge_lengths)

    #     raise ValueError(f"Unknown sizing specification: {take}. Available: min, max, avg")

    # def get_axis_vertex_pairs(self, axis: int) -> List[List[Vertex]]:
    #     """Returns 4 pairs of Vertex objects along given axis"""
    #     pairs = []

    #     for pair in c.AXIS_PAIRS[axis]:
    #         pair = [self.vertices[pair[0]], self.vertices[pair[1]]]

    #         if pair[0] == pair[1]:
    #             # omit vertices in the same spot; there is no edge anyway
    #             # (prisms/wedges/pyramids)
    #             continue

    #         if pair in pairs:
    #             # also omit duplicates
    #             continue

    #         pairs.append(pair)

    #     return pairs

    # def get_axis_from_pair(self, pair: List[Vertex]) -> Tuple[int, bool]:
    #     """returns axis index and orientation from a given pair of vertices;
    #     orientation is True if blocks are aligned or false when inverted.

    #     This can only be called after Mesh.write()"""
    #     riap = [pair[1], pair[0]]

    #     for i in range(3):
    #         pairs = self.get_axis_vertex_pairs(i)

    #         if pair in pairs:
    #             return i, True

    #         if riap in pairs:
    #             return i, False

    #     return None, None

    # @property
    # def patches(self) -> Dict[str, List[str]]:
    #     """Returns a dict of patches, for example:

    #     patches = {
    #          'volute_rotating': ['left', 'top' ],
    #          'volute_walls': ['bottom'],
    #     }"""
    #     # TODO: set type and other patch properties in mesh.boundary
    #     # TODO: test
    #     pdict = {}

    #     for orient, side in self.sides.items():
    #         if side.patch is not None:
    #             if side.patch not in pdict:
    #                 pdict[side.patch] = []

    #             pdict[side.patch].append(orient)

    #     return pdict