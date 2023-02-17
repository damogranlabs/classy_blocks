import dataclasses

from typing import List

from classy_blocks.data.division import Division

@dataclasses.dataclass
class Chop:
    """Stores divisions for a given pair of points (a.k.a. block edge)"""
    corner_1:int
    corner_2:int

    divisions:List[Division] = dataclasses.field(default_factory=list)

    def add_division(self, **kwargs) -> None:
        """Creates a Division object from args and adds it to the list"""
        self.divisions.append(Division(**kwargs))
