import numpy as np

from tests.fixtures.data import DataTestCase

from classy_blocks.data.block_data import BlockData

def data2pos(data:BlockData):
    """Creates a 2d numpy array from a list of Point objects
    from BlockData"""
    return np.array([point.pos for point in data.points])

class BlockDataTests(DataTestCase):
    def test_translate(self):
        """Translate BlockData object"""
        displacement = np.array([1, 1, 1])

        bd = self.get_single_data(0)
        original_points = data2pos(bd)

        bd.translate(displacement)
        translated_points = data2pos(bd)

        np.testing.assert_array_almost_equal(
            translated_points,
            original_points + displacement
        )