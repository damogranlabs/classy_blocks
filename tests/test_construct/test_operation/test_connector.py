import unittest

import numpy as np

from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.operations.connector import Connector


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        # basic box, center at origin
        self.box_1 = Box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        # a 'nicely' positioned box
        self.box_2 = self.box_1.copy().rotate(np.pi / 2, [0, 0, 1], [0, 0, 0]).translate([2, 0, 0])
        # an ugly box a.k.a. border case
        # self.box_3

    def test_create_normal(self):
        _ = Connector(self.box_1, self.box_2)

    def test_create_inverted(self):
        _ = Connector(self.box_2, self.box_1)

    def test_direction(self):
        connector = Connector(self.box_1, self.box_2)

        box_vector = self.box_2.center - self.box_1.center
        connector_vector = connector.top_face.center - connector.bottom_face.center

        self.assertGreater(np.dot(box_vector, connector_vector), 0)

    def test_direction_inverted(self):
        connector = Connector(self.box_2, self.box_1)

        box_vector = self.box_2.center - self.box_1.center
        connector_vector = connector.top_face.center - connector.bottom_face.center

        self.assertLess(np.dot(box_vector, connector_vector), 0)
