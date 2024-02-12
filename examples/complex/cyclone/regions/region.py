import abc
from typing import List, Set

from geometry import geometry

import classy_blocks as cb
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.util import functions as f


class Region(abc.ABC):
    """A logical unit of the mesh that can be constructed independently"""

    line_clamps: Set[int] = set()
    radial_clamps: Set[int] = set()
    plane_clamps: Set[int] = set()
    free_clamps: Set[int] = set()

    geo = geometry

    @property
    @abc.abstractmethod
    def elements(self) -> List[Operation]:
        """Entities of any type to be added to the mesh"""

    @abc.abstractmethod
    def chop(self) -> None:
        """Chops elements to cells"""

    def get_line_clamps(self, mesh):
        clamps: Set[cb.ClampBase] = set()

        for index in self.line_clamps:
            vertex = mesh.vertices[index]

            delta_x = f.vector(self.geo.r["inlet"] / 2, 0, 0)

            clamp = cb.LineClamp(vertex, vertex.position, vertex.position + delta_x, (-100, 100))
            clamps.add(clamp)

        return clamps

    def get_free_clamps(self, mesh: cb.Mesh) -> Set[cb.ClampBase]:
        clamps: Set[cb.ClampBase] = set()

        for index in self.free_clamps:
            vertex = mesh.vertices[index]
            clamps.add(cb.FreeClamp(vertex))

        return clamps

    def get_plane_clamps(self, mesh: cb.Mesh) -> Set[cb.ClampBase]:
        clamps: Set[cb.ClampBase] = set()

        for index in self.plane_clamps:
            vertex = mesh.vertices[index]
            clamp = cb.PlaneClamp(vertex, vertex.position, [0, 0, 1])
            clamps.add(clamp)

        return clamps

    def get_radial_clamps(self, mesh: cb.Mesh) -> Set[cb.ClampBase]:
        clamps: Set[cb.ClampBase] = set()

        for index in self.radial_clamps:
            vertex = mesh.vertices[index]
            clamp = cb.RadialClamp(vertex, [0, 0, 0], [0, 0, 1])
            clamps.add(clamp)

        return clamps

    def get_clamps(self, mesh: cb.Mesh) -> Set[cb.ClampBase]:
        """Returns a list of clamps to be used for mesh optimization"""
        clamps = self.get_line_clamps(mesh)
        clamps.update(self.get_radial_clamps(mesh))
        clamps.update(self.get_plane_clamps(mesh))
        clamps.update(self.get_free_clamps(mesh))

        return clamps

    def project(self) -> None:  # noqa: B027
        """Projections to geometry, if needed"""

    def set_patches(self):  # noqa: B027
        """Set pathes, if appropriate"""
