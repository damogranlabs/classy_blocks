from typing import List

from classy_blocks.base.exceptions import UndefinedGradingsError
from classy_blocks.items.block import Block


class BlockList:
    """Handling of the 'blocks' part of blockMeshDict, along with
    count/grading propagation and whatnot"""

    def __init__(self) -> None:
        self.blocks: List[Block] = []

    def add(self, block: Block) -> None:
        """Add blocks"""
        block.index = len(self.blocks)

        self.blocks.append(block)
        self.update_neighbours(block)

    def update_neighbours(self, new_block: Block) -> None:
        """Find and assign neighbours of a given block entry"""
        for block in self.blocks:
            if block == new_block:
                continue

            block.add_neighbour(new_block)
            new_block.add_neighbour(block)

    def propagate_gradings(self):
        """Copy references to gradings from defined blocks to their neighbours"""
        # a riddle similar to sudoku, keep traversing
        # and copying counts until there's no undefined blocks left
        undefined_blocks = set(range(len(self.blocks)))

        while len(undefined_blocks) > 0:
            updated = False

            for i in undefined_blocks:
                block = self.blocks[i]

                if block.is_defined:
                    undefined_blocks.remove(i)
                    updated = True
                    break

                updated = block.copy_grading() or updated

            if not updated:
                # All of the blocks were traversed and none was updated;
                # it won't get any better with next iterations
                break

        if len(undefined_blocks) > 0:
            # gather more detailed information about non-defined blocks:
            message = "Blocks with non-defined counts: \n"
            for i in list(undefined_blocks):
                message += f"{i}: "
                for axis in (0, 1, 2):
                    message += str(self.blocks[i].axes[axis].grading.count) + " "
                message += "\n"

            raise UndefinedGradingsError(message)

    def clear(self) -> None:
        """Removes created blocks"""
        self.blocks.clear()

    @property
    def description(self) -> str:
        """Outputs a list of blocks to be inserted directly into blockMeshDict"""
        out = "blocks\n(\n"

        for block in self.blocks:
            out += block.description

        out += ");\n\n"

        return out
