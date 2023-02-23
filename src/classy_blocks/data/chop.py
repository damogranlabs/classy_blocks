import dataclasses

from typing import Optional

from classy_blocks.types import ChopTakeType

@dataclasses.dataclass
class Chop:
    """A single 'chop' represents a division in Grading object;
    user-provided arguments are stored in this object and used
    for creation of Gradient object"""
    length_ratio:float = 1
    start_size:Optional[float] = None
    c2c_expansion:Optional[float] = None
    count:Optional[int] = None
    end_size:Optional[float] = None
    total_expansion:Optional[float] = None
    invert:bool = False
    take:ChopTakeType = 'avg'
