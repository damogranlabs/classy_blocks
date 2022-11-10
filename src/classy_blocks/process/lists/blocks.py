from typing import List, Set

from classy_blocks.define.grading import Grading
from classy_blocks.define.block import Block

class BlockList:
    """ Handling of the 'blocks' part of blockMeshDict, along with
    count/grading propagation and whatnot """
    def __init__(self):
        self.blocks:List[Block] = []

        # a list, parallel to self.blocks, containing each block's neighbours (also a list of indexes)
        # will be assigned by Mesh.write()
        self.neighbours:List[Set[int]] = []

        # grading for each block, translated from 'chops'
        # will be assigned by Mesh.write()
        self.gradings:List[List[Grading]] = []

    def add(self, block:Block) -> None:
        """ Adds a block to this list """
        block.index = len(self.blocks)

        self.neighbours.append(set())
        self.gradings.append([Grading(), Grading(), Grading()])
        self.blocks.append(block)

    def collect_neighbours(self) -> None:
        """ Generates a list that defines neighbours for each blocks;
        speeds up count/grading propagation since only this 'neighbours' list
        has to be traversed """
        for i, block in enumerate(self.blocks):
            for axis in range(3):
                axis_pairs = block.get_axis_vertex_pairs(axis)

                for i_cnd, candidate in enumerate(self.blocks):
                    if candidate == block:
                        continue

                    for p in axis_pairs:
                        cnd_axis, _ = candidate.get_axis_from_pair(p)
                        if cnd_axis is not None:
                            # the 'candidate' shares the same edge or face
                            self.neighbours[i].add(i_cnd)

    def convert_gradings(self) -> None:
        """Feeds block.chops to Grading objects"""
        # A list of 3 gradings for each block
        # now is the time to set counts
        for i_block, block in enumerate(self.blocks):
            for i_axis in range(3):
                params = block.chops[i_axis]
                grading = self.gradings[i_block][i_axis]

                if len(params) < 1:
                    continue

                block_size = block.get_size(i_axis, take=params[0].pop("take", "avg"))
                grading.set_block_size(block_size)

                for p in params:
                    grading.add_division(**p)

    def copy_grading(self, block_index, axis) -> bool:
        """Finds a block that shares an edge with given block
        and copies its grading along that axis"""
        # there are 4 pairs of vertices on specified axis:
        match_pairs = self.blocks[block_index].get_axis_vertex_pairs(axis)

        # first, find a block in mesh that shares one of the
        # edges in match_pairs:
        for nei_index in self.neighbours[block_index]:
            nei_block = self.blocks[nei_index]

            for p in match_pairs:
                b_axis, direction = nei_block.get_axis_from_pair(p)
                if b_axis is not None:
                    # b.get_axis_from_pair() returns axis index in
                    # the block we want to copy from;
                    if self.gradings[nei_index][b_axis].is_defined:
                        # this block's count/grading is defined on this axis
                        # so we can (must) copy it
                        self.gradings[block_index][axis] = self.gradings[nei_index][b_axis].copy(invert=not direction)

                        return True

        return False

    def propagate_gradings(self) -> None:
        """ Propagates cell count and grading from connected blocks """
        # a riddle similar to sudoku, keep traversing
        # and copying counts until there's no undefined blocks left
        undefined_blocks = list(range(len(self.blocks)))
        updated = False

        while len(undefined_blocks) > 0:
            for i in undefined_blocks:
                for axis in range(3):
                    if not self.gradings[i][axis].is_defined:
                        updated = self.copy_grading(i, axis) or updated

                if self.is_grading_defined(i):
                    undefined_blocks.remove(i)
                    updated = True
                    break

            if not updated:
                # a whole iteration went around without an update;
                # next iterations won't get any better
                break

            updated = False

        if len(undefined_blocks) > 0:
            # gather more detailed information about non-defined blocks:
            message = "Blocks with non-defined counts: \n"
            for i in list(undefined_blocks):
                message += f"\t{i}:"
                for axis in range(3):
                    message += f" {self.gradings[i][axis].count}"
                message += "\n"
            message += "\n"

            raise Exception(message)

    def assemble(self) -> None:
        """Collects all data neede to assemble the blocks() list in blockMeshDict."""
        self.convert_gradings()
        self.collect_neighbours()
        self.propagate_gradings()

    def is_grading_defined(self, i) -> bool:
        """Returns True if grading is defined in all dimensions"""
        return all(g.is_defined for g in self.gradings[i])

    def output(self) -> str:
        """ Outputs a list of blocks to be inserted directly into blockMeshDict """
        blist = "blocks\n(\n"

        for i, block in enumerate(self.blocks):
            # hex definition
            blist += "\thex "
            # vertices
            blist += " ( " + " ".join(str(v.mesh_index) for v in block.vertices) + " ) "

            # cellZone
            blist += block.cell_zone

            # number of cells
            grading = self.gradings[i]

            blist += f" ({grading[0].count} {grading[1].count} {grading[2].count}) "
            # grading
            blist += f" ({grading[0].grading} {grading[1].grading} {grading[2].grading})"

            # add a comment with block index
            blist += f" // {i} {block.description}\n"

        blist += ");\n\n"

        return blist

    def __getitem__(self, index):
        return self.blocks[index]

    def __len__(self):
        return len(self.blocks)
