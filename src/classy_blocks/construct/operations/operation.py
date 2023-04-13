from typing import List, Optional, Dict, Union, TypeVar, Set

import numpy as np

from classy_blocks.types import AxisType, NPPointType, PointType, VectorType, OrientType

from classy_blocks.base.additive import AdditiveBase

from classy_blocks.items.frame import Frame

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.edges import EdgeData, Line, Project
from classy_blocks.grading.chop import Chop

from classy_blocks.util import constants

OperationT = TypeVar("OperationT", bound="Operation")


class Operation(AdditiveBase):
    """A base class for all single-block operations
    (Box, Loft, Revolve, Extrude, Wedge)."""

    SIDES_MAP = ("front", "right", "back", "left")  # connects orients/indexes of side faces/edges

    def __init__(self, bottom_face: Face, top_face: Face):
        # An Operation only holds data to create the block;
        # actual object is created by Mesh when it is added.

        # Top and bottom faces contain all points:
        self.faces: Dict[OrientType, Face] = {}
        self.faces["bottom"] = bottom_face
        self.faces["top"] = top_face

        # side edges, patch names and projections
        # are held within the other four faces but
        # points and edges are duplicated; therefore
        # some care must be taken when getting information
        # from an Operation:
        #  - 8 points at corners from top and bottom faces
        #  - edges on top and bottom from corresponding faces
        #  - side edges are the first edge in each of left/right/front/back faces
        #  - projected faces are not duplicated, each face holds its own info
        # see self.set_* methods and properties
        for orient in self.SIDES_MAP:
            self.faces[orient] = Face([self.point_array[i] for i in constants.FACE_MAP[orient]])

        # instructions for cell counts and gradings
        self.chops: Dict[AxisType, List[Chop]] = {0: [], 1: [], 2: []}

        # optionally, put the block in a cell zone
        self.cell_zone = ""

    def add_side_edge(self, corner_1: int, edge_data: EdgeData) -> None:
        """Add an edge between two vertices at the same
        corner of the lower and upper face (index and index+4 or vice versa)."""
        assert corner_1 < 4, "corner_1 must be an index to a bottom Vertex (0...3)"

        # TODO: TEST
        orient = self.SIDES_MAP[corner_1]
        self.faces[orient].edges[0] = edge_data

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
            self.faces[orient].patch_name = name

    def project_side(self, side: OrientType, geometry: str, edges: bool = False, points: bool = False) -> None:
        """Project given side to named geometry;

        Args:
        - side: 'bottom', 'top', 'front', 'back', 'left', 'right';
            only
            the sketch from blockMesh documentation:
            https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility
            bottom, top: faces from which the Operation was created
            front: along first edge of a face
            back: opposite front
            right: along second edge of a face
            left: opposite right
        - geometry: name of predefined geometry (add separately to Mesh object)
        - edges:if True, all edges belonging to this side will also be projected"""
        self.faces[side].project(geometry, edges, points)

    def project_edge(self, corner_1: int, corner_2: int, geometry: Union[str, List[str]]) -> None:
        """Project an edge to a surface or an intersection of two surfaces"""
        # TODO: TEST
        corner_set = {corner_1, corner_2}

        side_pairs = [set(pair) for pair in constants.AXIS_PAIRS[2]]
        if corner_set in side_pairs:
            # that's a side edge
            corner_start = min(corner_1, corner_2)
            self.faces[self.SIDES_MAP[corner_start]].edges[0] = Project(geometry)
        
        top_pairs = [set(pair) for pair in constants.AXIS_PAIRS[1]]
        if corner_set in top_pairs:
            


    def project_corner(self, corner: int, geometry: Union[str, List[str]]) -> None:
        """Project the vertex at given corner (local index 0...7) to a single
        surface or an intersection of multiple surface. WIP according to
        https://github.com/OpenFOAM/OpenFOAM-10/blob/master/src/meshTools/searchableSurfaces/searchableSurfacesQueries/searchableSurfacesQueries.H
        """

    def set_cell_zone(self, cell_zone: str) -> None:
        """Assign a cellZone to this block."""
        self.cell_zone = cell_zone

    @property
    def operations(self: OperationT) -> List[OperationT]:
        return [self]

    @property
    def point_array(self) -> NPPointType:
        """Returns 8 points from which this operation is created"""
        return np.concatenate((self.faces["bottom"].point_array, self.faces["top"].point_array))

    @property
    def frame(self) -> Frame:
        """Returns a Frame with edges as its beams"""
        frame: Frame[EdgeData] = Frame()

        for i, data in enumerate(self.faces["bottom"].edges):
            if data is not None:
                frame.add_beam(i, (i + 1) % 4, data)

        for i, data in enumerate(self.faces["top"].edges):
            if data is not None:
                frame.add_beam(4 + i, 4 + (i + 1) % 4, data)

        for i, orient in enumerate(self.SIDES_MAP):
            data = self.faces[orient].edges[0]
            if data is not None:
                frame.add_beam(i, i + 4, data)

        return frame

    def get_patches_at_corner(self, corner: int) -> Set[str]:
        """Returns patch names at given corner (up to 3)"""
        patches = set()

        for side, patch in self.patch_names.items():
            if corner in constants.FACE_MAP[side]:
                patches.add(patch)

        return patches
