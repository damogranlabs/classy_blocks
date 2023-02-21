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
        self.update_gradings()

    def update_neighbours(self, new_block:Block) -> None:
        """Find and assign neighbours of a given block entry"""
        for block in self.blocks:
            if block == new_block:
                continue

            block.add_neighbour(new_block)
            new_block.add_neighbour(block)
    
    def update_gradings(self) -> None:
        """Updates gradings on all blocks"""
        # first, pull in all neighbours' gradings at coincident wires;
        # then, distribute gradings within blocks;
        # repeat the procedure until it makes no more changes
        changed = False

        while changed:
            for block in self.blocks:
                changed = block.frame.pull_gradings() or changed
            
            for block in self.blocks:
                for i in (0, 1, 2):
                    changed = block.frame.broadcast_grading(i) or changed

    @property
    def description(self) -> str:
        """ Outputs a list of blocks to be inserted directly into blockMeshDict """
        out = "blocks\n(\n"

        for block in self.blocks:
            out += block.description
            out += '\n'

        out += ");\n\n"

        return out
