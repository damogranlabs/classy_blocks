from typing import List, Dict, Union, TypeVar, Optional

import numpy as np

from classy_blocks.types import AxisType, NPPointType, OrientType, ProjectToType

from classy_blocks.base.element import ElementBase
from classy_blocks.util.frame import Frame
from classy_blocks.util.tools import edge_map
from classy_blocks.construct.edges import Line
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.edges import EdgeData, Project
from classy_blocks.grading.chop import Chop
from classy_blocks.util.constants import SIDES_MAP

from classy_blocks.util import constants

OperationT = TypeVar("OperationT", bound="Operation")


class Operation(ElementBase):
    """A base class for all single-block operations
    (Box, Loft, Revolve, Extrude, Wedge)."""

    # connects orients/indexes of side faces/edges

    def __init__(self, bottom_face: Face, top_face: Face):
        self.bottom_face = bottom_face
        self.top_face = top_face

        self.side_edges: List[EdgeData] = [Line(), Line(), Line(), Line()]
        self.side_projects: List[Optional[str]] = [None, None, None, None]
        self.side_patches: List[Optional[str]] = [None, None, None, None]

        # instructions for cell counts and gradings
        self.chops: Dict[AxisType, List[Chop]] = {0: [], 1: [], 2: []}

        # optionally, put the block in a cell zone
        self.cell_zone = ""

    def add_side_edge(self, corner_1: int, edge_data: EdgeData) -> None:
        """Add an edge between two vertices at the same
        corner of the lower and upper face (index and index+4 or vice versa)."""
        assert corner_1 < 4, "corner_1 must be an index to a bottom Vertex (0...3)"

        self.side_edges[corner_1] = edge_data

    def chop(self, axis: AxisType, **kwargs) -> None:
        """Chop the operation (count/grading) in given axis:
        0: along first edge of a face
        1: along second edge of a face
        2: between faces / along operation path

        Kwargs: see arguments for Chop object"""
        self.chops[axis].append(Chop(**kwargs))

    def unchop(self, axis: AxisType) -> None:
        """Removed existing chops from an operation
        (comes handy after copying etc.)"""
        self.chops[axis] = []

    def project_corner(self, corner: int, geometry: ProjectToType) -> None:
        """Project the vertex at given corner (local index 0...7) to a single
        surface or an intersection of multiple surface. WIP according to
        https://github.com/OpenFOAM/OpenFOAM-10/blob/master/src/meshTools/searchableSurfaces/searchableSurfacesQueries/searchableSurfacesQueries.H
        """
        # bottom and top faces define operation's points
        if corner > 3:
            self.top_face.points[corner - 4].project(geometry)
        else:
            self.bottom_face.points[corner].project(geometry)

    def project_edge(self, corner_1: int, corner_2: int, geometry: ProjectToType) -> None:
        """Replace an edge between given corners with a Projected one"""
        # decide where the required edge sits
        loc = edge_map[corner_1][corner_2]

        edge = Project(geometry)

        # bottom or top face?
        if loc.side == "bottom":
            self.bottom_face.edges[loc.start_corner] = edge
            return

        if loc.side == "top":
            self.top_face.edges[loc.start_corner] = edge
            return

        # sides
        self.side_edges[loc.start_corner] = edge

    def project_side(self, side: OrientType, geometry: str, edges: bool = False, points: bool = False) -> None:
        """Project given side to named geometry;

        Args:
        - side: 'bottom', 'top', 'front', 'back', 'left', 'right';
            the sketch from blockMesh documentation:
            https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility
            bottom, top: faces from which the Operation was created
            front: along first edge of a face
            back: opposite front
            right: along second edge of a face
            left: opposite right
        - geometry: name of predefined geometry (add separately to Mesh object)
        - edges:if True, all edges belonging to this side will also be projected"""
        # TODO: TEST with other sides
        if side == "bottom":
            self.bottom_face.project(geometry, edges, points)
            return

        if side == "top":
            self.top_face.project(geometry, edges, points)
            return

        index_1 = self.get_index_from_side(side)
        index_2 = (index_1 + 1) % 4

        self.side_projects[index_1] = geometry

        if edges:
            self.side_edges[index_1] = Project(geometry)
            self.side_edges[index_2] = Project(geometry)

            self.top_face.edges[index_1] = Project(geometry)
            self.bottom_face.edges[index_1] = Project(geometry)

        if points:
            for face in (self.top_face, self.bottom_face):
                for point_index in (index_1, index_2):
                    face.points[point_index].project(geometry)

    def set_patch(self, sides: Union[OrientType, List[OrientType]], name: str) -> None:
        """Assign a patch to given side of the block;

        Args:
        - side: 'bottom', 'top', 'front', 'back', 'left', 'right',
            a single value or a list of sides; names correspond to position in
            the sketch from blockMesh documentation:
            https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility
            bottom, top: faces from which the Operation was created
            front: along first edge of a face
            back: opposite front
            right: along second edge of a face
            left: opposite right
        - name: the name that goes into blockMeshDict

        Use mesh.set_patch_* methods to change other properties (type and other settings)"""
        if not isinstance(sides, list):
            sides = [sides]

        for orient in sides:
            if orient == "bottom":
                self.bottom_face.patch_name = name

            elif orient == "top":
                self.top_face.patch_name = name

            else:
                self.side_patches[self.get_index_from_side(orient)] = name

    def set_cell_zone(self, cell_zone: str) -> None:
        """Assign a cellZone to this block."""
        self.cell_zone = cell_zone

    @property
    def parts(self):
        return [self.bottom_face, self.top_face, *self.side_edges]

    @property
    def point_array(self) -> NPPointType:
        """Returns 8 points from which this operation is created"""
        return np.concatenate((self.bottom_face.point_array, self.top_face.point_array))

    @property
    def center(self):
        return np.average(self.point_array, axis=0)

    def get_patches_at_corner(self, corner: int) -> set:
        """Returns patch names at given corner (up to 3)"""
        patches = set()

        # 1st patch: from top or bottom face
        if corner < 4:
            patches.add(self.bottom_face.patch_name)
        else:
            patches.add(self.top_face.patch_name)

        # 2nd and 3rd patch: from the next an previous side at that corner
        index = corner % 4

        patches.add(self.side_patches[index])
        patches.add(self.side_patches[(index + 3) % 4])

        # clean up Nones
        patches.discard(None)
        return patches

    def get_face(self, side: OrientType) -> Face:
        """Get or create a Face on specified side of the Operation"""
        if side == "bottom":
            return self.bottom_face

        if side == "top":
            return self.top_face

        # for other sides, faces must be created
        # but edges and projections need not be copied
        return Face([self.point_array[i] for i in constants.FACE_MAP[side]])

    @property
    def patch_names(self) -> Dict[OrientType, str]:
        """Returns patches names on sides where they are specified"""
        patch_names: Dict[OrientType, str] = {}

        def add(orient, name):
            if name is not None:
                patch_names[orient] = name

        add("bottom", self.bottom_face.patch_name)
        add("top", self.top_face.patch_name)

        for index, orient in enumerate(SIDES_MAP):
            add(orient, self.side_patches[index])

        return patch_names

    @property
    def edges(self) -> Frame[EdgeData]:
        """Returns a Frame with edges as its beams"""
        edges = Frame[EdgeData]()

        for i, data in enumerate(self.bottom_face.edges):
            edges.add_beam(i, (i + 1) % 4, data)

        for i, data in enumerate(self.top_face.edges):
            edges.add_beam(i + 4, (i + 1) % 4 + 4, data)

        for i, data in enumerate(self.side_edges):
            edges.add_beam(i, i + 4, data)

        return edges

    @staticmethod
    def get_index_from_side(side: OrientType) -> int:
        """Returns index of edges/patches/projections from given orient"""
        assert side in SIDES_MAP, "Use self.top_face()/self.bottom_face() for actions on top and bottom face"

        return SIDES_MAP.index(side)
