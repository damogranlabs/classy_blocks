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

    def update_lengths(self) -> None:
        """Update lengths on grading objects"""
        # Grading on each wire was specified with length 0;
        # after edges have been added to blocks, they now have a proper
        # value to work with
        for block in self.blocks:
            for wire in block.wire_list:
                wire.update()

    def __hash__(self):
        return id(self)
