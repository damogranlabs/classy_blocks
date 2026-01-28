import abc

from geometry import geometry

import classy_blocks as cb
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.util import functions as f


class Region(abc.ABC):
    """A logical unit of the mesh that can be constructed independently"""

    line_clamps: tuple[int, ...] = ()
    radial_clamps: tuple[int, ...] = ()
    plane_clamps: tuple[int, ...] = ()
    free_clamps: tuple[int, ...] = ()

    geo = geometry

    @property
    @abc.abstractmethod
    def elements(self) -> list[Operation]:
        """Entities of any type to be added to the mesh"""

    def get_line_clamps(self, mesh):
        clamps: set[cb.ClampBase] = set()

        for index in self.line_clamps:
            vertex = mesh.vertices[index]

            delta_x = f.vector(self.geo.r["inlet"] / 2, 0, 0)

            clamp = cb.LineClamp(vertex.position, vertex.position, vertex.position + delta_x, (-100, 100))
            clamps.add(clamp)

        return clamps

    def get_free_clamps(self, mesh: cb.Mesh) -> set[cb.ClampBase]:
        clamps: set[cb.ClampBase] = set()

        for index in self.free_clamps:
            vertex = mesh.vertices[index]
            clamps.add(cb.FreeClamp(vertex.position))

        return clamps

    def get_plane_clamps(self, mesh: cb.Mesh) -> set[cb.ClampBase]:
        clamps: set[cb.ClampBase] = set()

        for index in self.plane_clamps:
            vertex = mesh.vertices[index]
            clamp = cb.PlaneClamp(vertex.position, vertex.position, [0, 0, 1])
            clamps.add(clamp)

        return clamps

    def get_radial_clamps(self, mesh: cb.Mesh) -> set[cb.ClampBase]:
        clamps: set[cb.ClampBase] = set()

        for index in self.radial_clamps:
            vertex = mesh.vertices[index]
            clamp = cb.RadialClamp(vertex.position, [0, 0, 0], [0, 0, 1])
            clamps.add(clamp)

        return clamps

    def get_clamps(self, mesh: cb.Mesh) -> set[cb.ClampBase]:
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
