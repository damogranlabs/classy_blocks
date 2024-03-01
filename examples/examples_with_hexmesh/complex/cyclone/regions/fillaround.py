import parameters as params
from regions.region import Region

import classy_blocks as cb
from classy_blocks.util import functions as f


class FillAround(Region):
    """A ring that leaves out lofts that interfere with loft_1 and loft_2,
    then connects them to form a complete 'ring'."""

    def __init__(self, loft_1: cb.Loft, loft_2: cb.Loft):
        self.loft_1 = loft_1
        self.loft_2 = loft_2

        center = [0, 0, loft_1.center[2]]

        inner_face = loft_1.get_closest_face(center)
        outer_face = loft_1.get_closest_face(100 * loft_1.center)
        bottom_face = loft_1.get_closest_face([0, 0, -100])
        top_face = loft_1.get_closest_face([0, 0, 100])

        z_bottom = min([p[2] for p in bottom_face.point_array])
        z_top = max([p[2] for p in top_face.point_array])
        r_inner = min([f.to_polar(p)[0] for p in inner_face.point_array])
        r_outer = max([f.to_polar(p)[0] for p in outer_face.point_array])

        # if one takes 11 segments, then after omitting first and last 3,
        # 12 blocks are obtained together with skirt and other pieces
        ring = cb.ExtrudedRing([0, 0, z_bottom], [0, 0, z_top], [0, -r_outer, z_bottom], r_inner, n_segments=11)

        shell = ring.shell[1:-4]

        # connect first and last lofts
        connector_1 = cb.Connector(loft_1, shell[0])
        connector_2 = cb.Connector(shell[-1], loft_2)

        self._operations = [connector_1, *ring.shell[1:-4], connector_2]

    def chop(self):
        self.elements[1].chop(2, start_size=params.BULK_SIZE)
        self.elements[1].chop(0, end_size=params.BL_THICKNESS, c2c_expansion=1 / params.C2C_EXPANSION)

    @property
    def elements(self):
        return self._operations

    @property
    def connector_1(self):
        return self._operations[0]

    @property
    def connector_2(self):
        return self._operations[-1]

    def project(self):
        self.connector_1.project_side("left", "body", True, True)
        self.connector_2.project_side("right", "body", True, True)
