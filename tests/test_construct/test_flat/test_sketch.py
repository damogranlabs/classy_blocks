import unittest

from classy_blocks.construct.flat.quad import find_neighbours, get_fixed_points


class MappedSketchTests(unittest.TestCase):
    def test_find_neighbours(self):
        # A random blocking (quadding)
        quads = [(1, 2, 7, 6), (2, 3, 4, 7), (7, 4, 5, 6), (0, 1, 6, 8)]

        neighbours = find_neighbours(quads)

        self.assertDictEqual(
            neighbours,
            {
                0: {8, 1},
                1: {0, 2, 6},
                2: {1, 3, 7},
                3: {2, 4},
                4: {3, 5, 7},
                5: {4, 6},
                6: {8, 1, 5, 7},
                7: {2, 4, 6},
                8: {0, 6},
            },
        )

    def test_fixed_points(self):
        # Monocylinder, core is quads[0]
        quads = quads = [
            (0, 1, 2, 3),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]

        fixed_points = get_fixed_points(quads)

        self.assertSetEqual(
            fixed_points,
            {4, 5, 6, 7},
        )
