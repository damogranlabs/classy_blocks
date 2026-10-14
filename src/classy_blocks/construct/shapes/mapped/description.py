import functools
import json
import os
from dataclasses import dataclass

import numpy as np

from classy_blocks.cbtyping import IndexType, NPPointListType
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface
from classy_blocks.optimize.cell import HexCell


@dataclass
class MappedShapeDescription:
    # a list of points defining the whole shape
    points: NPPointListType
    # point connectivity; each block is represented
    # by 8-tuple (same vertex ordering as in blockMesh)
    hexas: list[IndexType]
    # faces, defined here, belong to a patch
    patches: dict[str, list[IndexType]]
    # patch faces will be mapped to specified surfaces
    surfaces: dict[str, TriangulatedSurface]

    # TODO: morphing and optimization data

    @functools.cached_property
    def inner(self) -> list[int]:
        """Indexes of points that blocks use but don't lie on the outside of the shape"""
        sides = np.sort(np.asarray(self.hexas)[:, HexCell.side_indexes].reshape(-1, 4), axis=1)
        unique, counts = np.unique(sides, axis=0, return_counts=True)

        return np.setdiff1d(np.unique(self.hexas), unique[counts == 1]).tolist()

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
            {name: TriangulatedSurface.from_file(os.path.join(base, rel)) for name, rel in content["surfaces"].items()},
        )
