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
        """Chains ordered loops (as returned by get_cross_sections) into a single
        path. Each loop keeps its own point order; only its direction is chosen.
        Starting from 'far_point', the loop with an end nearest to the current
        path end is appended next, flipped if needed, so 'far_point' anchors
        where the path starts and therefore its direction.

        Consecutive points within 'tol' of each other along the resulting path
        are merged (dropped): this removes coincident points such as closed-loop
        closing vertices and thins over-refined sections. Raise 'tol' to coarsen;
        the default merges only effectively-identical points."""
        remaining = [np.asarray(loop) for loop in loops]
        chained: list[NPPointListType] = [np.empty((0, 3))]
        end = np.asarray(far_point)

        while remaining:
            ends = np.array([[loop[0], loop[-1]] for loop in remaining])
            index, flip = np.unravel_index(np.argmin(np.linalg.norm(ends - end, axis=2)), ends.shape[:2])
            loop = remaining.pop(index)
            chained.append((loop, loop[::-1])[flip])
            end = chained[-1][-1]

        path = np.concatenate(chained)
        keep = np.ones(len(path), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > tol
        return path[keep]
