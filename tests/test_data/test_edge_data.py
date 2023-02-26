import numpy as np

from tests.fixtures.data import DataTestCase

from classy_blocks.data.point import Point
from classy_blocks.data.edge_data import EdgeData

class BlockDataTests(DataTestCase):
    def test_translate(self):
        """Translate BlockData object"""
        displacement = np.array([1, 1, 1])
    
        ed = EdgeData(0, 1, 'arc', [[0.5, 0.2, 0]])
        original_pos = np.copy(ed.points[0].pos)

        ed.translate(displacement)
        translated_pos = ed.points[0].pos

        np.testing.assert_array_almost_equal(
            original_pos + displacement,
            translated_pos
        )
