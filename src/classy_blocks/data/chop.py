import dataclasses
from typing import List

from classy_blocks.types import AxisType, ChopTakeType
from classy_blocks.grading.grading import Grading
from classy_blocks.grading.division import Division

@dataclasses.dataclass
class ChopBase:
    """Collects user-specified divisions for chopping/grading blocks"""
    divisions:List[Division] = dataclasses.field(default_factory=list)

    def add_division(self, **kwargs) -> None:
        """Creates a Division object from args and adds it to the list"""
        self.divisions.append(Division(**kwargs))

    def get_grading(self, length:float) -> Grading:
        """Creates a Grading object from specified chops"""
        g = Grading(length)
        for div in self.divisions:
            g.add_division(div)

        return g

@dataclasses.dataclass(kw_only=True)
class AxisChop(ChopBase):
    """Stores chops for all 4 edges in given direction (axis)"""
    axis:AxisType
    take:ChopTakeType

@dataclasses.dataclass(kw_only=True)
class EdgeChop(ChopBase):
    """Stores divisions for a given pair of points (a.k.a. block edge)"""
    corner_1:int
    corner_2:int
