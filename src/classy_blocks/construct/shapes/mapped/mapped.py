from typing import Callable

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator
from trimesh.proximity import closest_point
from trimesh.registration import procrustes

from classy_blocks.base.exceptions import MappedShapeCreationError
from classy_blocks.cbtyping import MatrixType, NPPointListType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.shapes.mapped.association import Kind, Label, associate
from classy_blocks.construct.shapes.mapped.description import MappedShapeDescription
from classy_blocks.construct.shapes.mapped.features import Features
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.clamps.curve import CurveClamp
from classy_blocks.optimize.clamps.free import FreeClamp
from classy_blocks.optimize.clamps.surface import TriangulatedSurfaceClamp
from classy_blocks.optimize.optimizer import ShapeOptimizer

# corners have no clamp: they stay where they are
CLAMPS: dict[Kind, Callable] = {
    Kind.FREE: lambda point, _: FreeClamp(point),
    Kind.SURFACE: TriangulatedSurfaceClamp,
    Kind.CURVE: CurveClamp,
}


class MappedShape(Shape):
    """Similar to a MappedSketch, this object takes a list of points in space
    and their connectivity (list of indexes that form each hexahedron)
    to construct an arbitrary, user-defined Shape."""

    def __init__(self, description: MappedShapeDescription):
        self.description = description
        # point indexes of patches that have a surface of the same name
        self.fitted = {
            name: np.unique(faces) for name, faces in description.patches.items() if name in description.surfaces
        }

        features = Features.from_surfaces(description.surfaces)

        points = np.array(description.points)
        if self.fitted:  # without surfaces, the description is taken as it is
            points = trimesh.transform_points(points, self._register())
            points = self._conform(points, associate(points, description.patches, features))

        self.labels = associate(points, description.patches, features)

        self._lofts = [Loft(Face(points[hexa[:4]]), Face(points[hexa[4:]])) for hexa in description.hexas]

    def _register(self, iterations: int = 100) -> MatrixType:
        """Finds a matrix transform that moves initial description's points
        closest to the actual provided triangulated surfaces."""
        sources = np.concatenate([self.description.points[idx] for idx in self.fitted.values()])
        matrix = np.eye(4)

        for _ in range(iterations):
            moved = trimesh.transform_points(sources, matrix)
            targets = np.concatenate(
                [
                    closest_point(self.description.surfaces[name].mesh, moved[i : i + len(idx)])[0]
                    for (name, idx), i in zip(self.fitted.items(), np.cumsum([0, *map(len, self.fitted.values())]))
                ]
            )
            matrix = procrustes(sources, targets, reflection=False, scale=True)[0]

        return matrix

    def _conform(self, points: NPPointListType, labels: list[Label]) -> NPPointListType:
        """Moves points to their targets: surfaces, curves or corners;
        the free ones follow by a smooth interpolation of their displacements."""
        errors = [f"Point {i}: {label.error}" for i, label in enumerate(labels) if label.error]
        if errors:
            raise MappedShapeCreationError("Description doesn't match the surfaces", "\n\t".join(errors))

        pinned = np.array([i for i, label in enumerate(labels) if label.target is not None])
        snapped = np.array([labels[i].target.get_closest_point(points[i]) for i in pinned])  # type: ignore[union-attr]
        warp = RBFInterpolator(points[pinned], snapped - points[pinned], kernel="thin_plate_spline")

        return points + warp(points)

    @property
    def operations(self):
        return self._lofts

    @property
    def grid(self):
        # Grid is irrelevant for arbitrary mapped shapes
        return [self.operations]

    @property
    def points(self) -> NPPointListType:
        """Current positions of description's points"""
        points = np.array(self.description.points)
        points[np.ravel(self.description.hexas)] = np.concatenate([loft.point_array for loft in self._lofts])

        return points

    def optimize(self, report: bool = True) -> bool:
        """Improves blocking quality by moving points along what they are stuck to;
        points on the outside without a surface stay where they are."""
        optimizer = ShapeOptimizer(self.operations, report=report)
        points = self.points
        pinned = [i for i, label in enumerate(self.labels) if label.kind in (Kind.SURFACE, Kind.CURVE)]

        for i in [*self.description.inner, *pinned]:
            clamp: ClampBase = CLAMPS[self.labels[i].kind](points[i], self.labels[i].target)
            optimizer.add_clamp(clamp)

        return optimizer.optimize(tolerance=1e-5)

    @classmethod
    def from_file(cls, path: str) -> "MappedShape":
        description = MappedShapeDescription.load(path)
        return cls(description)
