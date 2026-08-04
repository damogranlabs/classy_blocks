from classy_blocks.items.block import Block
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lookup.point_registry import HexPointRegistry


class BlockList:
    """Handling of the 'blocks' part of blockMeshDict"""

    def __init__(self) -> None:
        self.blocks: list[Block] = []

    def add(self, block: Block) -> None:
        """Add blocks"""
        self.blocks.append(block)

    def _locate_wires(self, registry: HexPointRegistry, vertex_list: VertexList) -> None:
        """Tells every wire where it is and whether it lies within a face-merged patch.

        The registry knows where points are; vertex numbering does not, because
        face merging duplicates vertices on slave patches. Wires must know both
        before neighbours and coincidents can be established."""
        slave_patches = {entry.vertex.index: frozenset(entry.patches) for entry in vertex_list.duplicated}
        no_patches: frozenset[str] = frozenset()

        for block in self.blocks:
            # blocks are added in the same order as cells are registered
            canonical = registry.cell_addressing[block.index]

            for wire in block.wire_list:
                wire.locate(
                    (canonical[wire.corners[0]], canonical[wire.corners[1]]),
                    slave_patches.get(wire.vertices[0].index, no_patches)
                    & slave_patches.get(wire.vertices[1].index, no_patches),
                )

    def update_neighbours(self, registry: HexPointRegistry, vertex_list: VertexList) -> None:
        """Find and assign neighbours of a given block entry"""
        self._locate_wires(registry, vertex_list)

        for block in self.blocks:
            neighbour_indexes = registry.find_cell_neighbours(block.index)

            for i in neighbour_indexes:
                block.add_neighbour(self.blocks[i])

    def __hash__(self):
        return id(self)
