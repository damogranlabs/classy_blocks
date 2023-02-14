from typing import List

from classy_blocks.items.block import Block

from classy_blocks.util import constants

class BlockList:
    """ Handling of the 'blocks' part of blockMeshDict, along with
    count/grading propagation and whatnot """
    def __init__(self):
        self.blocks:List[Block] = []

    def add(self, block:Block) -> None:
        """Add blocks"""
        self.blocks.append(block)
        self.update_neighbours(block)

    def update_neighbours(self, new_block):
        """Find and assign neighbours of a given block entry"""
        for block in self.blocks:
            if block == new_block:
                continue

            block.add_neighbour(new_block)
            new_block.add_neighbour(block)
    
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

    # def is_grading_defined(self, i) -> bool:
    #     """Returns True if grading is defined in all dimensions"""
    #     return all(g.is_defined for g in self.gradings[i])

    @property
    def description(self) -> str:
        """ Outputs a list of blocks to be inserted directly into blockMeshDict """
        out = "blocks\n(\n"

        for block in self.blocks:
            out += block.description
            out += '\n'

        out += ");\n\n"

        return out
