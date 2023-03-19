import abc

from typing import Dict, List

from classy_blocks.base.additive import AdditiveBase

class Shape(AdditiveBase, abc.ABC):
    """A collection of Operations that form a predefined
    parametric shape"""
    def set_cell_zone(self, cell_zone:str) -> None:
        """Sets cell zone for all blocks in this shape"""
        for op in self.operations:
            op.set_cell_zone(cell_zone)
