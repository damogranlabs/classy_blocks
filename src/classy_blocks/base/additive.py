import abc

from typing import List

from classy_blocks.items.block import Block

class AdditiveBase(abc.ABC):
    """A base class for any entity that can be added to mesh using
    mesh.add(); with all the machinery required to do that"""
    @property
    @abc.abstractmethod
    def blocks(self) -> List[Block]:
        """Blocks to be added to mesh"""
