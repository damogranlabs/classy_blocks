import abc

from typing import List

from classy_blocks.construct.operations.operation import Operation

class AdditiveBase(abc.ABC):
    """A base class for any entity that can be added to mesh using
    mesh.add(); with all the machinery required to do that"""
    @property
    @abc.abstractmethod
    def operations(self) -> List[Operation]:
        """Operations to be added to mesh"""
