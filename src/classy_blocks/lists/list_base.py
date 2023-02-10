import abc

from classy_blocks.base import ObjectBase

class ListBase(abc.ABC):
    """Common methods for all 'lists'"""

    @abc.abstractmethod
    def add(self, items:List[ObjectBase]) -> None:
        """Extract relevant data from added objects"""