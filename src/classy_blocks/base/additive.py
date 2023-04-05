import abc

from typing import Dict, List, TYPE_CHECKING

from classy_blocks.base.transformable import TransformableBase

if TYPE_CHECKING:
    from classy_blocks.construct.operations.operation import Operation


class AdditiveBase(TransformableBase):
    """A base class for any entity that can be added to mesh using
    mesh.add(); with all the machinery required to do that"""

    @property
    @abc.abstractmethod
    def operations(self) -> List["Operation"]:  # TODO: sort out (Operation is a child of AdditiveBase?)
        """A list of operations to be added to mesh"""

    @property
    def geometry(self) -> Dict[str, List[str]]:
        """Some shapes need to project faces to a
        geometry, defined ad-hoc at creation time;
        the mesh will look into this property to add those"""
        return {}
