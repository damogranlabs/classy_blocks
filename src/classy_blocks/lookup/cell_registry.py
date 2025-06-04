from typing import Dict, List, Set

from classy_blocks.cbtyping import IndexType
from classy_blocks.util import functions as f


class CellRegistry:
    def __init__(self, addressing: List[IndexType]):
        self.addressing = addressing

        # build a map of cells that belong to each point;
        # first, find the maximum point index
        max_index = max(f.flatten_2d_list(addressing))

        self.near_cells: Dict[int, Set[int]] = {i: set() for i in range(max_index + 1)}

        for i_cell, cell_indexes in enumerate(self.addressing):
            for i_point in cell_indexes:
                self.near_cells[i_point].add(i_cell)

    def get_near_cells(self, point_index: int) -> Set[int]:
        return self.near_cells[point_index]
