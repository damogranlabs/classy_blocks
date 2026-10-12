import os
from typing import Optional

import numpy as np
import trimesh

from classy_blocks.cbtyping import NPPointListType, NPPointType, PointType, VectorType
from classy_blocks.construct.surfaces.surface import SurfaceBase


class TriangulatedSurface(SurfaceBase):
    """A triangulated surface, backed by a trimesh object; typically loaded
    from an STL file and used as a fixed reference for curves and snapping."""

    def __init__(self, mesh: trimesh.Trimesh, path: Optional[str] = None) -> None:
        self.mesh = mesh
        # absolute path of the file this surface was loaded from, if any
        self.path = path

    @classmethod
    def from_file(cls, path: str, solid: Optional[str] = None) -> "TriangulatedSurface":
        """Loads a surface from a file (STL/OBJ/...); trimesh detects the format.
        From a file with several solids (a multi-solid STL) it takes the named one, otherwise the whole file."""
        meshes = trimesh.load_scene(path).geometry
        mesh = meshes[solid] if solid in meshes else trimesh.load_mesh(path)

        return cls(mesh, os.path.abspath(path))

    def get_closest_point(self, point: PointType) -> NPPointType:
        closest, _, _ = trimesh.proximity.closest_point(self.mesh, [np.asarray(point)])
        return closest[0]

    def get_cross_sections(self, point: PointType, normal: VectorType) -> list[NPPointListType]:
        section = self.mesh.section(plane_origin=np.asarray(point), plane_normal=np.asarray(normal))
        if section is None:
            return []

        return [np.asarray(entity.discrete(section.vertices)) for entity in section.entities]
