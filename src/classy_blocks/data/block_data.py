"""Contains all data that user can specify about a Block."""
import warnings

from typing import List, Union, Dict

from classy_blocks.types import PointListType, AxisType, OrientType


from classy_blocks.data.data_object_base import DataObjectBase

from classy_blocks.data.side import Side
from classy_blocks.data.chop import Chop

from classy_blocks.util import constants

class BlockData(DataObjectBase):
    """User-provided data for a block"""
    def __init__(self, points: PointListType):
        super().__init__(points)

        # cell counts and grading
        self.axis_chops:Dict[AxisType, List[Chop]] = { 0: [], 1: [], 2: [] }
        # TODO: edge chops
        #self.edge_chops:List[EdgeChop] = []

        # Side objects define patch names and projections
        self.sides = {o:Side(o) for o in constants.FACE_MAP}

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.comment = ""

    def chop(self, axis: AxisType, **kwargs:Union[str, float, int, bool]) -> None:
        """Set block's cell count/size and grading for a given direction/axis.
        Exactly two of the following keyword arguments must be provided:

        :Keyword Arguments:
        * *count:
            number of cells;
            Optionally, this can be the only provided argument;
            in that case c2c_expansion will be set to 1.
        * *start_size:
            size of the first cell (last if invert==True)
        * *end_size:
            size of the last cell
        * *c2c_expansion:
            cell-to-cell expansion ratio
        * *total_expansion:
            ratio between first and last cell size

        :Optional keyword arguments:
        * *invert:
            reverses grading if True
        * *take:
            must be 'min', 'max', or 'avg'; takes minimum or maximum edge
            length for block size calculation, or average of all edges in given direction.
            With multigrading only the first 'take' argument is used, others are copied.
        * *length_ratio:
            in case the block is graded using multiple gradings, specify
            length of current division; see
            https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading;
            Multiple gradings are specified by multiple calls to .chop() with
            the same 'axis' parameter."""
        self.axis_chops[axis].append(Chop(**kwargs))

    def set_patch(self,
        orients: Union[OrientType, List[OrientType]],
        patch_name: str,
        patch_type:str='patch'
    ) -> None:
        """assign one or more block sides (constants.FACE_MAP) to a chosen patch name;
        if type is not specified, it will becom 'patch'"""
        if isinstance(orients, str):
            orients = [orients]

        for orient in orients:
            if self.sides[orient].patch_name is not None:
                warnings.warn(f"Replacing patch {self.sides[orient].patch_name} with {patch_name}")

            self.sides[orient].patch_name = patch_name
            self.sides[orient].patch_type = patch_type

    # def project_face(self, orient:OrientType, geometry: str, edges: bool = False) -> None:
    #     """Assign one or more block faces (self.face_map)
    #     to be projected to a geometry (defined in Mesh)"""
    #     assert orient in constants.FACE_MAP

    #     self.sides[orient].project = geometry

    #     if edges:
    #         for i in range(4):
    #             self.add_edge(i, (i + 1) % 4, 'project', geometry)

    @property
    def blocks(self) -> List['BlockData']:
        return [self]
