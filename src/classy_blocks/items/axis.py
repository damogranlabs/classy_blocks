import dataclasses

from typing import List

from classy_blocks.types import AxisType

from classy_blocks.data.chop import Chop
from classy_blocks.items.wire import Wire


@dataclasses.dataclass
class Axis:
    """One of block axes, numbered 0, 1, 2, and the relevant data"""
    index:AxisType
    wires:List[Wire]
    chops:List[Chop]

    @property
    def lengths(self) -> List[float]:
        """Returns length for each wire of this axis; to be used
        for grading calculation"""
        return [wire.edge.length for wire in self.wires]
