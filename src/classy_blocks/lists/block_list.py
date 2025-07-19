from classy_blocks.base.exceptions import UndefinedGradingsError
from classy_blocks.items.block import Block
from classy_blocks.lookup.point_registry import HexPointRegistry


class BlockList:
    """Handling of the 'blocks' part of blockMeshDict, along with
    count/grading propagation and whatnot"""

    def __init__(self) -> None:
        self.blocks: list[Block] = []

    def add(self, block: Block) -> None:
        """Add blocks"""
        self.blocks.append(block)

    def update_neighbours(self, navigator: HexPointRegistry) -> None:
        """Find and assign neighbours of a given block entry"""
        for block in self.blocks:
            neighbour_indexes = navigator.find_cell_neighbours(block.index)

            for i in neighbour_indexes:
                block.add_neighbour(self.blocks[i])

    def assemble(self) -> None:
        self.update_lengths()
        self.grade()
        self.check_definitions()
        self.check_consistency()

    def update_lengths(self) -> None:
        """Update lengths on grading objects"""
        # Grading on each wire was specified with length 0;
        # after edges have been added to blocks, they now have a proper
        # value to work with
        for block in self.blocks:
            for wire in block.wire_list:
                wire.update()

    def grade(self) -> None:
        undefined_blocks = set(self.blocks)

        while len(undefined_blocks) > 0:
            removed: set[Block] = set()

            for block in undefined_blocks:
                if block.is_defined:
                    continue

                block.grade()

                if block.is_defined:
                    removed.add(block)

            if len(removed) == 0:
                # All of the blocks were traversed and none was updated;
                # it won't get any better with next iterations
                break

            undefined_blocks -= removed

    def check_definitions(self) -> None:
        undefined_blocks: list[Block] = []

        for block in self.blocks:
            if not block.is_defined:
                undefined_blocks.append(block)

        if len(undefined_blocks) > 0:
            # gather more detailed information about non-defined blocks:
            message = "Blocks with non-defined counts: \n"
            for block in undefined_blocks:
                message += f"{block.index}: "
                for axis in (0, 1, 2):
                    message += str(block.axes[axis].count) + " "
                message += "\n"

            raise UndefinedGradingsError(message)

    def check_consistency(self) -> None:
        """Check that all wires of each block axis have the same count;
        also check that all coincident wires have the same length and grading"""
        for block in self.blocks:
            block.check_consistency()
