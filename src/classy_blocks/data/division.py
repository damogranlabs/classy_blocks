import dataclasses
from typing import Optional

@dataclasses.dataclass
class Division:
    """Stores chopping parameters for a given division of
    an (optionally) multigraded block"""
    # arguments
    length_ratio:float = 1
    start_size:Optional[float] = None
    c2c_expansion:Optional[float] = None
    count:Optional[int] = None
    end_size:Optional[float] = None
    total_expansion:Optional[float] = None
    invert:bool = False