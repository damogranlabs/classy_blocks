from typing import Optional

import numpy as np
import trimesh

from classy_blocks.base.element import ElementBase
from classy_blocks.cbtyping import NPPointType, PointType, VectorType
from classy_blocks.construct.curves.curve import CurveBase
from classy_blocks.construct.curves.interpolated import LinearInterpolatedCurve
from classy_blocks.construct.point import Point
from classy_blocks.construct.surfaces.surface import SurfaceBase
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class RevolvedSurface(SurfaceBase, ElementBase):
    """A surface of revolution: a profile curve swept around an axis,
    parametrized as p = f(t, a) with curve parameter t and revolution angle a.

    The revolution axis is stored as two points (origin and origin + unit axis)
    so translate/rotate/scale/mirror compose through ElementBase for free."""

    def __init__(
        self,
        curve: CurveBase,
        axis: VectorType,
        origin: PointType = (0, 0, 0),
        angle_bounds: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        self.curve = curve
        self._origin = Point(origin)
        self._axis_tip = Point(np.asarray(origin, dtype=float) + f.unit_vector(axis))
        self.angle_bounds = angle_bounds

    @property
    def parts(self):
        return [self.curve, self._origin, self._axis_tip]

    @property
    def center(self) -> NPPointType:
        return self._origin.position

    @property
    def origin(self) -> NPPointType:
        return self._origin.position

    @property
    def axis(self) -> NPPointType:
        return f.unit_vector(self._axis_tip.position - self._origin.position)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.curve.bounds, self.angle_bounds)

    def get_point(self, param: float, angle: float) -> NPPointType:
        """Point on the surface at curve parameter 'param' and revolution 'angle'.
        'param' is validated by the curve; 'angle' is applied as given and is NOT clamped to angle_bounds."""
        return f.rotate(self.curve.get_point(param), angle, self.axis, self.origin)

    def _reference_radial(self) -> NPPointType:
        """Unit radial vector to the discretized curve point farthest from the axis."""
        axis = self.axis
        relative = self.curve.discretize() - self.origin
        radial = relative - np.outer(relative @ axis, axis)
        radii = np.linalg.norm(radial, axis=1)
        farthest = int(np.argmax(radii))
        return radial[farthest] / radii[farthest]

    def get_closest_params(self, point: PointType) -> tuple[float, float]:
        """Finds the parameters (curve parameter, revolution angle) at which this surface
        is the closest to the given point; feed them to get_point() to obtain that point.
        The angle is clamped to angle_bounds, the parameter to the curve's bounds."""
        axis = self.axis
        origin = self.origin
        reference = self._reference_radial()

        relative = np.asarray(point, dtype=float) - origin
        radial = relative - (relative @ axis) * axis

        # signed azimuth of the query about the axis, measured from the meridian;
        # arctan2 is magnitude-invariant so unnormalized 'radial' is fine, and an
        # on-axis query (radial == 0) yields angle 0 with no special-casing.
        angle = float(np.arctan2(np.cross(reference, radial) @ axis, reference @ radial))
        angle = float(np.clip(angle, self.angle_bounds[0], self.angle_bounds[1]))

        in_plane = f.rotate(point, -angle, axis, origin)
        return self.curve.get_closest_param(in_plane), angle

    def get_closest_point(self, point: PointType) -> NPPointType:
        return self.get_point(*self.get_closest_params(point))

    @classmethod
    def from_mesh(
        cls,
        mesh: trimesh.Trimesh,
        axis: VectorType,
        anchor: PointType,
        origin: Optional[PointType] = None,
        angle_bounds: tuple[float, float] = (-np.pi, np.pi),
    ) -> "RevolvedSurface":
        """Extracts a cross-section profile from a triangulated surface of revolution
        and builds a RevolvedSurface from it.

        'anchor' is a single off-axis point: defines the cutting plane and side.
        Its position along the axis is irrelevant; the profile is always ordered from the minimum-axis end upward."""
        axis_unit = f.unit_vector(axis)
        origin = mesh.centroid if origin is None else np.asarray(origin, dtype=float)
        assert origin is not None
        anchor = np.asarray(anchor, dtype=float)

        offset = anchor - origin
        radial = offset - (offset @ axis_unit) * axis_unit
        if f.norm(radial) < TOL:
            raise ValueError("RevolvedSurface.from_mesh: 'anchor' must lie off the revolution axis")

        perp = f.unit_vector(radial)
        normal = f.unit_vector(np.cross(axis_unit, radial))

        loops = TriangulatedSurface(mesh).get_cross_sections(origin, normal)
        if not loops:
            raise ValueError("RevolvedSurface.from_mesh: cut produced no profile (plane misses the mesh)")

        points = np.concatenate(loops)
        kept = points[(points - origin) @ perp >= 0]
        if len(kept) == 0:
            raise ValueError("RevolvedSurface.from_mesh: no profile points on the anchor's side of the axis")

        far_point = origin - f.norm(mesh.extents) * axis_unit
        profile = SurfaceBase.sort_points([kept], far_point=far_point)
        return cls(LinearInterpolatedCurve(profile), axis_unit, origin, angle_bounds)

    @classmethod
    def from_file(
        cls,
        path: str,
        axis: VectorType,
        anchor: PointType,
        origin: Optional[PointType] = None,
        angle_bounds: tuple[float, float] = (-np.pi, np.pi),
    ) -> "RevolvedSurface":
        """Loads a mesh (STL/OBJ/...) via trimesh, then delegates to from_mesh."""
        return cls.from_mesh(trimesh.load_mesh(path), axis, anchor, origin, angle_bounds)
