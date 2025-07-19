import warnings
from typing import Union

import numpy as np

from classy_blocks.cbtyping import IndexType, NPPointListType, NPPointType, PointListType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class MappedSketch(Sketch):
    """A sketch that is created from predefined points.
    The points are connected to form quads which define Faces."""

    def __init__(self, positions: PointListType, quads: list[IndexType]):
        self._faces: list[Face] = []
        self.indexes = quads

        for quad in self.indexes:
            face = Face([positions[iq] for iq in quad])
            self._faces.append(face)

        self.add_edges()

    def update(self, positions: PointListType) -> None:
        """Update faces with updated positions"""
        for i, quad in enumerate(self.indexes):
            points = [positions[iq] for iq in quad]

            self.faces[i].update(points)

    def add_edges(self) -> None:
        """An optional method that will add edges to faces;
        use `sketch.faces` property to access them."""

    @property
    def faces(self):
        """A 'flattened' grid of faces"""
        return self._faces

    @property
    def grid(self):
        """Use a single-tier grid by default; override the method for more sophistication."""
        return [self.faces]

    @property
    def center(self) -> NPPointType:
        """Center of this sketch"""
        return np.average([face.center for face in self.faces], axis=0)

    @property
    def positions(self) -> NPPointListType:
        """Reconstructs positions back from faces, so they are always up-to-date,
        even after transforms"""
        indexes = list(np.array(self.indexes).flatten())
        max_index = max(indexes)
        all_points = f.flatten_2d_list([face.point_array.tolist() for face in self.faces])

        return np.array([all_points[indexes.index(i)] for i in range(max_index + 1)])

    def merge(self, other: Union[list["MappedSketch"], "MappedSketch"]):
        """Adds a sketch or list of sketches to itself.
        New faces and indexes are appended and all duplicate points are removed."""

        def merge_two_sketches(sketch_1: MappedSketch, sketch_2: MappedSketch) -> None:
            """Add sketch_2 to sketch_1"""

            # Check planes are oriented the same
            if not abs(f.angle_between(sketch_1.normal, sketch_2.normal)) < constants.TOL:
                print(f.angle_between(sketch_1.normal, sketch_2.normal) / np.pi, sketch_1.normal, sketch_2.normal)
                warnings.warn(
                    f"Sketch {sketch_2} with normal {sketch_2.normal} is not oriented as "
                    f"sketch {sketch_1} with normal {sketch_1.normal}",
                    stacklevel=1,
                )

            # All unique points
            sketch_1_pos = sketch_1.positions
            all_pos = np.asarray(
                [
                    *sketch_1_pos.tolist(),
                    *[
                        pos
                        for pos in sketch_2.positions
                        if all(np.linalg.norm(sketch_1_pos - pos.reshape((1, 3)), axis=1) >= constants.TOL)
                    ],
                ]
            )

            sketch_2_ind = np.asarray(sketch_2.indexes)
            # Change sketch_2 indexes to new position list.
            for i, face in enumerate(sketch_2.faces):
                for j, pos in enumerate(face.point_array):
                    sketch_2_ind[i, j] = np.argwhere(
                        np.linalg.norm(all_pos - pos.reshape((1, 3)), axis=1) < constants.TOL
                    )[0][0]

            # Append indexes and faces to sketch_1
            sketch_1.indexes = [*list(sketch_1.indexes), *sketch_2_ind.tolist()]
            sketch_1._faces = [*sketch_1.faces, *sketch_2.faces]

        # If list of sketches
        if isinstance(other, list):
            for o in other:
                merge_two_sketches(self, o)
        else:
            merge_two_sketches(self, other)
