import abc

from typing import Sequence, TypeVar

AdditiveT = TypeVar('AdditiveT', bound='AdditiveBase')

class AdditiveBase(abc.ABC):
    """A base class for any entity that can be added to mesh using
    mesh.add(); with all the machinery required to do that"""
    @property
    @abc.abstractmethod
    def operations(self:AdditiveT) -> Sequence[AdditiveT]:
        """A list of operations to be added to mesh"""
