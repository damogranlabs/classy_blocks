import abc

from typing import Dict, List


class AdditiveBase:
    """A base class for any entity that can be added to mesh using
    mesh.add(); with all the machinery required to do that"""

    @property
    @abc.abstractmethod
    def operations(self) -> List:
        """A list of operations to be added to mesh"""

    @property
    def geometry(self) -> Dict[str, List[str]]:
        """Some shapes need to project faces to a
        geometry, defined ad-hoc at creation time;
        the mesh will look into this property to add those"""
        return {}
