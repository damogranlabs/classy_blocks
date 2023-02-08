from typing import List

from classy_blocks.data.block import BlockData
from classy_blocks.process.items.block_ops import BlockOps
from classy_blocks.process.lists.vertex_list import VertexList
from classy_blocks.process.items.edge_ops import EdgeOps
from classy_blocks.process.lists.edge_list import EdgeList

from classy_blocks.util import constants

class BlockList:
    """ Handling of the 'blocks' part of blockMeshDict, along with
    count/grading propagation and whatnot """
    def __init__(self):
        # added during mesh building
        self.blocks:List[BlockData] = []
        # generated during compilation
        self.ops:List[BlockOps] = []

    def add(self, blocks:List[BlockData]) -> None:
        """Add blocks"""
        self.blocks += blocks

    def generate_ops(self, vertex_list:VertexList) -> None:
        """Creates BlockOps objects with proper Vertex objects
        and everything else that belongs to them"""
        for block in self.blocks:
            vertices = [vertex_list.find(point) for point in block.points]
            self.ops.append(BlockOps(block, vertices, len(self.ops)))

    def collect_edges(self, edge_list:EdgeList) -> None:
        """Collects and assigns edges for each op"""
        for op in self.ops:
            for pair in constants.EDGE_PAIRS:
                vertex_1 = op.vertices[pair[0]]
                vertex_2 = op.vertices[pair[1]]

                try:
                    # find an edgebetween the vertices
                    # and add it to this op
                    edge = edge_list.find(vertex_1, vertex_2)
                    op.edges.append(edge)
                except RuntimeError:
                    # there's no such edge in the mesh;
                    # no biggie
                    continue

    # def get_neighbours(self, block) -> List[BlockEntry]:
    #     """Find and assign neighbours of a given block entry"""
    #     for axis in range(3):
    #         axis_pairs = block.get_axis_vertex_pairs(axis)

    #         for i_cnd, candidate in enumerate(self.blocks):
    #             if candidate == block:
    #                 continue

    #             for p in axis_pairs:
    #                 cnd_axis, _ = candidate.get_axis_from_pair(p)
    #                 if cnd_axis is not None:
    #                     # the 'candidate' shares the same edge or face
    #                     self.neighbours[i].add(i_cnd)



    # def collect_neighbours(self) -> None:
    #     """ Generates a list that defines neighbours for each blocks;
    #     speeds up count/grading propagation since only this 'neighbours' list
    #     has to be traversed """
    #     for i, block in enumerate(self.blocks):
    
    # def copy_grading(self, block_index, axis) -> bool:
    #     """Finds a block that shares an edge with given block
    #     and copies its grading along that axis"""
    #     # there are 4 pairs of vertices on specified axis:
    #     match_pairs = self.blocks[block_index].get_axis_vertex_pairs(axis)

    #     # first, find a block in mesh that shares one of the
    #     # edges in match_pairs:
    #     for nei_index in self.neighbours[block_index]:
    #         nei_block = self.blocks[nei_index]

    #         for p in match_pairs:
    #             b_axis, direction = nei_block.get_axis_from_pair(p)
    #             if b_axis is not None:
    #                 # b.get_axis_from_pair() returns axis index in
    #                 # the block we want to copy from;
    #                 if self.gradings[nei_index][b_axis].is_defined:
    #                     # this block's count/grading is defined on this axis
    #                     # so we can (must) copy it
    #                     self.gradings[block_index][axis] = self.gradings[nei_index][b_axis].copy(invert=not direction)

    #                     return True

    #     return False

    # def propagate_gradings(self) -> None:
    #     """ Propagates cell count and grading from connected blocks """
    #     # a riddle similar to sudoku, keep traversing
    #     # and copying counts until there's no undefined blocks left
    #     undefined_blocks = list(range(len(self.blocks)))
    #     updated = False

    #     while len(undefined_blocks) > 0:
    #         for i in undefined_blocks:
    #             for axis in range(3):
    #                 if not self.gradings[i][axis].is_defined:
    #                     updated = self.copy_grading(i, axis) or updated

    #             if self.is_grading_defined(i):
    #                 undefined_blocks.remove(i)
    #                 updated = True
    #                 break

    #         if not updated:
    #             # a whole iteration went around without an update;
    #             # next iterations won't get any better
    #             break

    #         updated = False

    #     if len(undefined_blocks) > 0:
    #         # gather more detailed information about non-defined blocks:
    #         message = "Blocks with non-defined counts: \n"
    #         for i in list(undefined_blocks):
    #             message += f"\t{i}:"
    #             for axis in range(3):
    #                 message += f" {self.gradings[i][axis].count}"
    #             message += "\n"
    #         message += "\n"

    #         raise Exception(message)

    def assemble(self, vertex_list:VertexList, edge_list:EdgeList) -> None:
        """Collects all data needed to assemble the blocks() list in blockMeshDict."""
        self.generate_ops(vertex_list)
        self.collect_edges(edge_list)
        #self.convert_gradings()
        #self.collect_neighbours()
        #self.propagate_gradings()

    # def is_grading_defined(self, i) -> bool:
    #     """Returns True if grading is defined in all dimensions"""
    #     return all(g.is_defined for g in self.gradings[i])

    # def output(self) -> str:
    #     """ Outputs a list of blocks to be inserted directly into blockMeshDict """
    #     blist = "blocks\n(\n"

    #     for i, block in enumerate(self.blocks):
    #         # hex definition
    #         blist += "\thex "
    #         # vertices
    #         blist += " ( " + " ".join(str(v.mesh_index) for v in block.vertices) + " ) "

    #         # cellZone
    #         blist += block.cell_zone

    #         # number of cells
    #         grading = self.gradings[i]

    #         blist += f" ({grading[0].count} {grading[1].count} {grading[2].count}) "
    #         # grading
    #         blist += f" ({grading[0].grading} {grading[1].grading} {grading[2].grading})"

    #         # add a comment with block index
    #         blist += f" // {i} {block.description}\n"

    #     blist += ");\n\n"

    #     return blist

    # def __getitem__(self, index):
    #     return self.hexas[index]

    # def __len__(self):
    #     return len(self.hexas)
