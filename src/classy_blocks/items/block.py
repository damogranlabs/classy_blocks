from typing import List

from classy_blocks.data.block import BlockData
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edge.base import Edge
from classy_blocks.grading import Grading

class Block:
    """Further operations on blocks"""
    def __init__(self, data:BlockData, index:int, vertices:List[Vertex], edges:List[Edge]):
        self.data = data
        self.index = index

        self.vertices = vertices
        self.edges = edges

        # TODO: convert
        self.gradings:List[Grading] = []

        # TODO: a separate list/object/something
        # self.neighbours:List[Block] = []

    # def convert_gradings(self) -> None:
    #     """Feeds block.chops to Grading objects"""
    #     # A list of 3 gradings for each block
    #     # now is the time to set counts
    #     for i_block, block in enumerate(self.blocks):
    #         for i_axis in range(3):
    #             params = block.chops[i_axis]
    #             grading = self.gradings[i_block][i_axis]

    #             if len(params) < 1:
    #                 continue

    #             block_size = block.get_size(i_axis, take=params[0].pop("take", "avg"))
    #             grading.set_block_size(block_size)

    #             for p in params:
    #                 grading.add_division(**p)

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

    @property
    def description(self) -> str:
        """hex definition for blockMesh"""
        # TODO: test
        out = "\thex "

        # vertices
        out += " ( " + " ".join(str(v.index) for v in self.vertices) + " ) "

        # cellZone
        #out += block.cell_zone

        # number of cells
        #grading = self.gradings[i]
            
        #out += f" ({grading[0].count} {grading[1].count} {grading[2].count}) "
        # grading
        #out += f" ({grading[0].grading} {grading[1].grading} {grading[2].grading})"

        out += ' (10 10 10) simpleGrading (1 1 1) // description'

            # add a comment with block index
            #out += f" // {i} {block.description}\n"

        return out
