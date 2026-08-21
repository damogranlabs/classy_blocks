import abc

import numpy as np

from classy_blocks.cbtyping import NPPointListType, NPPointType, PointType
from classy_blocks.util.constants import TOL


class SurfaceBase(abc.ABC):
    """A queryable surface in 3D space; the surface analog of CurveBase.

    Transformation is intentionally not supported here: subclasses that need it
    (e.g. RevolvedSurface) mix in ElementBase themselves."""

    @abc.abstractmethod
    def get_closest_point(self, point: PointType) -> NPPointType:
        """Returns the point on the surface closest to the given point."""

    @staticmethod
    def sort_points(loops: list[NPPointListType], far_point: PointType, tol: float = TOL) -> NPPointListType:
        """Chains loops (as returned by get_cross_sections) into a single ordered
        path: start at the point closest to 'far_point', then repeatedly append
        the nearest not-yet-used point. 'far_point' anchors where the path starts
        and therefore its direction.

        Consecutive points within 'tol' of each other along the resulting path
        are merged (dropped): this removes coincident points such as closed-loop
        closing vertices and thins over-refined sections. Raise 'tol' to coarsen;
        the default merges only effectively-identical points."""
        points = np.concatenate([np.asarray(loop) for loop in loops]) if loops else np.empty((0, 3))
        if len(points) == 0:
            return points

        remaining = list(range(len(points)))
        start = int(np.argmin(np.linalg.norm(points - np.asarray(far_point), axis=1)))
        order = [remaining.pop(start)]

        while remaining:
            distances = np.linalg.norm(points[remaining] - points[order[-1]], axis=1)
            order.append(remaining.pop(int(np.argmin(distances))))

        path = points[order]
        keep = np.ones(len(path), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > tol
        return path[keep]
