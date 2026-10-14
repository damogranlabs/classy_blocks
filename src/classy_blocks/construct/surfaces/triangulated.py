import os
from typing import Optional

import numpy as np
import trimesh

from classy_blocks.cbtyping import NPPointListType, NPPointType, NPVectorType, PointType, VectorType
from classy_blocks.construct.surfaces.surface import SurfaceBase


class TriangulatedSurface(SurfaceBase):
    """A triangulated surface, backed by a trimesh object; typically loaded
    from an STL file and used as a fixed reference for curves and snapping."""

    def __init__(self, mesh: trimesh.Trimesh, path: Optional[str] = None) -> None:
        self.mesh = mesh
        # absolute path of the file this surface was loaded from, if any
        self.path = path

    @classmethod
    def from_file(cls, path: str) -> "TriangulatedSurface":
        """Loads a surface from a file (STL/OBJ/...); trimesh detects the format.
        Multi-solid meshes are currently not supported - the whole file is taken as a single solid."""
        return cls(trimesh.load_mesh(path), os.path.abspath(path))

    def get_closest_point(self, point: PointType) -> NPPointType:
        closest, _, _ = trimesh.proximity.closest_point(self.mesh, [np.asarray(point)])
        return closest[0]

    def get_normal(self, point: PointType) -> NPVectorType:
        """Normal of the triangle closest to the point"""
        _, _, triangle = trimesh.proximity.closest_point(self.mesh, [np.asarray(point)])
        return self.mesh.face_normals[triangle[0]]

    def get_cross_sections(self, point: PointType, normal: VectorType) -> list[NPPointListType]:
        section = self.mesh.section(plane_origin=np.asarray(point), plane_normal=np.asarray(normal))
        if section is None:
            return []

        return [np.asarray(entity.discrete(section.vertices)) for entity in section.entities]
