"""a test mesh:
3 blocks, extruded in z-direction
(these indexes refer to 'fl' and 'cl' variables,
not vertex.mesh_index)

 ^ y-axis
 |
 |   7---6
     | 2 |
 3---2---5
 | 0 | 1 |
 0---1---4   ---> x-axis

After adding the blocks, the following mesh indexes are
in mesh.vertices:

Bottom 'floor':

 ^ y-axis
 |
 |  13--12
     | 2 |
 3---2---9
 | 0 | 1 |
 0---1---8   ---> x-axis

Top 'floor':

 ^ y-axis
 |
 |  15--14
     | 2 |
 7---6--11
 | 0 | 1 |
 4---5--10   ---> x-axis"""
import unittest
import dataclasses

from typing import List

import numpy as np

from classy_blocks.construct import edges
from classy_blocks.grading.chop import Chop

fl: List[List[float]] = [  # points on the 'floor'; z=0
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0],  # 3
    [2, 0, 0],  # 4
    [2, 1, 0],  # 5
    [2, 2, 0],  # 6
    [1, 2, 0],  # 7
]
fl_indexes = [0, 1, 2, 3, 8, 9, 12, 13]

cl = [[p[0], p[1], 1] for p in fl]  # points on ceiling; z = 1
cl_indexes = [4, 5, 6, 7, 10, 11, 14, 15]


@dataclasses.dataclass
class TestOperationData:
    """to store predefined data for test block creation"""

    # points from which to create the block
    point_indexes: List[int]

    # edges; parameters correspond to block.add_edge() args
    edges: List = dataclasses.field(default_factory=list)

    # chop counts (for each axis, use None to not chop)
    chops: List[List[Chop]] = dataclasses.field(default_factory=lambda: [[], [], []])

    # calls to set_patch()
    patches: List = dataclasses.field(default_factory=list)

    # other thingamabobs
    description: str = ""
    cell_zone: str = ""

    @property
    def points(self):
        # to create vertices
        return np.array([fl[i] for i in self.point_indexes] + [cl[i] for i in self.point_indexes])

    @property
    def indexes(self):
        # to create vertices
        return [fl_indexes[i] for i in self.point_indexes] + [cl_indexes[i] for i in self.point_indexes]


test_data = [
    TestOperationData(
        point_indexes=[0, 1, 2, 3],
        edges=[  # edges
            [0, 1, edges.Arc([0.5, -0.25, 0])],
            [1, 2, edges.Spline([[1.1, 0.25, 0], [1.05, 0.5, 0], [1.1, 0.75, 0]])],
        ],
        chops=[  # chops
            [Chop(count=6)],
            [],
            [],
        ],
        patches=[["left", "inlet"], [["bottom", "top", "front", "back"], "walls", "wall"]],
        description="Test",
    ),
    TestOperationData(
        point_indexes=[1, 4, 5, 2],
        edges=[
            [3, 0, edges.Arc([0.5, -0.1, 1])],  # duplicated edge in block 2 that must not be included
            [0, 1, edges.Arc([0.5, 0, 0])],  # collinear point; invalid edge must be dropped
        ],
        chops=[
            [Chop(count=5)],
            [Chop(count=6)],
            [],
        ],
        patches=[
            [["bottom", "top", "right", "front"], "walls", "wall"],
        ],
    ),
    TestOperationData(
        point_indexes=[2, 5, 6, 7],
        chops=[
            [],
            [Chop(count=8)],
            [Chop(count=7)],
        ],
        patches=[["back", "outlet"], [["bottom", "top", "left", "right"], "walls"]],
    ),
]


class DataTestCase(unittest.TestCase):
    """Test case with ready-made block data"""

    @staticmethod
    def get_single_data(index: int) -> TestOperationData:
        """Returns a list of predefined blocks for testing"""
        return test_data[index]

    @staticmethod
    def get_all_data() -> List[TestOperationData]:
        """Returns all prepared block data"""
        return test_data

    @staticmethod
    def get_vertex_indexes(index: int) -> List[int]:
        """Indexes of the points used for block creation;
        will be used in tests to create Vertices manually"""
        return test_data[index].point_indexes
