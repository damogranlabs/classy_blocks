import numpy as np
import trimesh

from classy_blocks.cbtyping import NPPointListType, NPPointType, PointType, VectorType
from classy_blocks.construct.surfaces.surface import SurfaceBase


class TriangulatedSurface(SurfaceBase):
    """A triangulated surface, backed by a trimesh object; typically loaded
    from an STL file and used as a fixed reference for curves and snapping."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.mesh = mesh

    @classmethod
    def from_file(cls, path: str) -> "TriangulatedSurface":
        """Loads a surface from a file (STL/OBJ/...); trimesh detects the format."""
        return cls(trimesh.load_mesh(path))

    def get_closest_point(self, point: PointType) -> NPPointType:
        closest, _, _ = trimesh.proximity.closest_point(self.mesh, [np.asarray(point)])
        return closest[0]

    def get_cross_sections(self, point: PointType, normal: VectorType) -> list[NPPointListType]:
        section = self.mesh.section(plane_origin=np.asarray(point), plane_normal=np.asarray(normal))
        if section is None:
            return []

        return [np.asarray(loop) for loop in section.discrete]
