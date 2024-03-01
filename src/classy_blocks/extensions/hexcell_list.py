from typing import List

from classy_blocks.extensions.hexcell import HexCell


class HexCellList:
    """Handling of the HexCells part of hexMesh"""

    # this is simplified version of BlockList

    def __init__(self) -> None:
        self.hexcells: List[HexCell] = []

    def add(self, hexcell: HexCell) -> None:
        """Add cells"""
        hexcell.index = len(self.hexcells)
        self.hexcells.append(hexcell)
        # add cell index to the vertices of this cell
        for hexvertex in hexcell.hexvertices:
            hexvertex.cell_indices.append(hexcell.index)
        # self.update_neighbours(hexcell)

    def update_neighbours(self, new_cell: HexCell) -> None:
        """Find and assign neighbours of a given cell entry"""
        for hexcell in self.hexcells:
            if hexcell == new_cell:
                continue

            hexcell.add_neighbour(new_cell)
            new_cell.add_neighbour(hexcell)

    def clear(self) -> None:
        """Removes created cells"""
        self.hexcells.clear()
