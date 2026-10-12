import json
import os
from dataclasses import dataclass

import numpy as np

from classy_blocks.cbtyping import IndexType, PointListType
from classy_blocks.construct.surfaces.surface import SurfaceBase
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface


@dataclass
class MappedShapeDescription:
    # a list of points defining the whole shape
    points: PointListType
    # point connectivity; each block is represented
    # by 8-tuple (same vertex ordering as in blockMesh)
    hexas: list[IndexType]
    # faces, defined here, belong to a patch
    patches: dict[str, list[IndexType]]
    # patch faces will be mapped to specified surfaces
    surfaces: dict[str, SurfaceBase]

    # TODO: morphing and optimization data

    @classmethod
    def load(cls, path: str) -> "MappedShapeDescription":
        """Surface paths in the file are relative to the file itself"""
        base = os.path.dirname(os.path.abspath(path))
        with open(path) as handle:
            content = json.load(handle)

        return cls(
            np.asarray(content["points"], dtype=float).reshape(-1, 3),
            [[int(i) for i in hexa] for hexa in content["hexas"]],
            {name: [[int(i) for i in face] for face in faces] for name, faces in content["patches"].items()},
            {
                name: TriangulatedSurface.from_file(os.path.join(base, rel), name)
                for name, rel in content["surfaces"].items()
            },
        )
