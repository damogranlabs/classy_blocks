"""What a set of surfaces offers to stick points to: the surfaces themselves,
curves where two of them meet and corners where more of them meet"""

import functools
import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.sparse
import trimesh
from scipy.sparse.csgraph import connected_components, depth_first_order

from classy_blocks.cbtyping import NPPointListType, NPPointType, PointType
from classy_blocks.construct.curves.interpolated import LinearInterpolatedCurve
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface

RELATIVE_TOLERANCE = 1e-5  # of the surfaces' combined bounding-box diagonal

Names = frozenset[str]


@dataclass(frozen=True)
class Corner:
    """A point where three or more surfaces meet"""

    point: NPPointType

    def get_closest_point(self, _point: PointType) -> NPPointType:
        return self.point


Target = Union[TriangulatedSurface, LinearInterpolatedCurve, Corner]


def _on(surface: TriangulatedSurface, points: NPPointListType, tolerance: float) -> np.ndarray:
    """Which points lie on the surface"""
    if len(points) == 0:  # trimesh can't query no points at all
        return np.zeros(0, dtype=bool)

    _, distances, _ = trimesh.proximity.closest_point(surface.mesh, points)
    return distances < tolerance


def _nearer(point: NPPointType, one: Target, other: Target) -> Target:
    return min(one, other, key=lambda target: float(np.linalg.norm(target.get_closest_point(point) - point)))


@dataclass
class Shared:
    """Boundary edges of one surface that lie on another, as vertex index pairs into its vertices"""

    vertices: NPPointListType
    edges: np.ndarray

    @classmethod
    def find(cls, surface: TriangulatedSurface, other: TriangulatedSurface, tolerance: float) -> "Shared":
        mesh = surface.mesh
        boundary = mesh.edges_sorted[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        # with its middle too: an edge may span between two pieces of the other surface
        ends = mesh.vertices[boundary]
        probes = np.concatenate([ends, ends.mean(axis=1, keepdims=True)], axis=1)
        on = _on(other, probes.reshape(-1, 3), tolerance).reshape(-1, 3).all(axis=1)

        return cls(mesh.vertices, boundary[on])

    def polylines(self) -> list[NPPointListType]:
        """Points of each connected piece in order: from an end, or around a closed loop back to its start"""
        nodes, inverse = np.unique(self.edges, return_inverse=True)
        pairs = inverse.reshape(-1, 2)
        graph = scipy.sparse.coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(nodes), len(nodes))).tocsr()
        degrees = np.bincount(pairs.ravel(), minlength=len(nodes))
        _, piece = connected_components(graph, directed=False)

        def polyline(index: int) -> NPPointListType:
            members = np.flatnonzero(piece == index)
            start = members[np.argmin(degrees[members])]
            order = list(depth_first_order(graph, start, directed=False, return_predecessors=False))
            closed = int(degrees[start] == 2)

            return self.vertices[nodes[order + order[:closed]]]

        return [polyline(index) for index in np.unique(piece)]


@dataclass
class Features:
    surfaces: dict[str, TriangulatedSurface]
    # surfaces may meet in several places: each piece is its own curve or corner
    curves: dict[Names, list[LinearInterpolatedCurve]]
    corners: dict[Names, list[Corner]]

    @functools.cached_property
    def _by_names(self) -> dict[Names, Sequence[Target]]:
        surfaces: dict[Names, Sequence[Target]] = {
            frozenset([name]): [surface] for name, surface in self.surfaces.items()
        }

        return {**surfaces, **self.curves, **self.corners}

    def find(self, names: Names, point: PointType) -> Optional[Target]:
        """The surface, curve or corner where exactly the named surfaces meet;
        of several pieces, the one nearest to the point"""
        # reduce leaves a single candidate alone: a surface needs no costly distance
        return functools.reduce(functools.partial(_nearer, np.asarray(point)), self._by_names.get(names, [None]))

    @classmethod
    def from_surfaces(cls, surfaces: dict[str, TriangulatedSurface], tolerance: Optional[float] = None) -> "Features":
        tolerance = (
            tolerance or RELATIVE_TOLERANCE * trimesh.util.concatenate([s.mesh for s in surfaces.values()]).scale
        )

        # a pipe ending on a wall's interior only has a boundary on the pipe's side
        shared = {
            frozenset([name, other]): max(
                Shared.find(surfaces[name], surfaces[other], tolerance),
                Shared.find(surfaces[other], surfaces[name], tolerance),
                key=lambda found: len(found.edges),
            )
            for name, other in itertools.combinations(surfaces, 2)
        }
        curves = {
            names: [LinearInterpolatedCurve(polyline) for polyline in found.polylines()]
            for names, found in shared.items()
            if len(found.edges) > 0
        }

        gathered: dict[Names, list[NPPointType]] = defaultdict(list)
        for names, pieces in curves.items():
            points = np.concatenate([curve.series.points for curve in pieces])
            others = [name for name in surfaces if name not in names]
            on = (
                np.array([_on(surfaces[name], points, tolerance) for name in others])
                .reshape(len(others), len(points))
                .T
            )

            for point, row in zip(points, on):
                gathered[names | {name for name, hit in zip(others, row) if hit}].append(point)

        # every curve point lands somewhere; only those on further surfaces name a corner
        corners = {
            names: [Corner(point) for point in trimesh.grouping.group_distance(points, tolerance)[0]]
            for names, points in gathered.items()
            if len(names) > 2
        }

        return cls(surfaces, curves, corners)
