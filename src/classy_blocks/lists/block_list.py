from classy_blocks.items.block import Block
from classy_blocks.lookup.point_registry import HexPointRegistry


class BlockList:
    """Handling of the 'blocks' part of blockMeshDict"""

    def __init__(self) -> None:
        self.blocks: list[Block] = []

    def add(self, block: Block) -> None:
        """Add blocks"""
        self.blocks.append(block)

    def update_neighbours(self, registry: HexPointRegistry) -> None:
        """Find and assign neighbours of a given block entry"""
        for block in self.blocks:
            neighbour_indexes = registry.find_cell_neighbours(block.index)

            for i in neighbour_indexes:
                block.add_neighbour(self.blocks[i])

    def __hash__(self):
        return id(self)
