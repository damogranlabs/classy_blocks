import abc
from typing import List

class ObjectBase(abc.ABC):
    """An abstract class defining common properties for all
    blocks/operations/shapes/objects"""
    @property
    @abc.abstractmethod
    def blocks(self) -> List['ObjectBase']:
        """A list of single block (Block/Operation)
        or multiple blocks defining this object; to be added to Mesh"""
