import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.grading.autograding.params.approximator import Approximator, Piece


def get_coords(count, c2c_expansion, start_coord, start_size):
    expansions = np.cumprod(np.ones(count) * c2c_expansion)
    sizes = np.ones(count) * start_size * expansions
    coords = np.ones(count + 1) * start_coord
    coords[1:] += np.cumsum(sizes)

    return coords


class PieceTests(unittest.TestCase):
    def test_piece_coords_linear(self):
        given_coords = np.linspace(0.1, 1.1, num=10)
        piece = Piece(given_coords)

        np.testing.assert_almost_equal(given_coords, piece.get_chop_coords())

    @parameterized.expand(
        (
            (0.1, 1, 10),
            (0.1, 10, 10),
            (-10, -1, 10),
            (-1, -10, 10),
        )
    )
    def test_piece_coords_geom(self, start, end, count):
        given_coords = np.geomspace(start, end, num=count)
        piece = Piece(given_coords)

        np.testing.assert_almost_equal(given_coords, piece.get_chop_coords())

    def test_piece_coords_mixed_sign(self):
        given_coords = get_coords(10, 1.2, -0.1, 0.1)

        piece = Piece(given_coords)

        np.testing.assert_almost_equal(given_coords, piece.get_chop_coords())

    def test_piece_coords_flipped(self):
        given_coords = np.flip(get_coords(10, 1.2, -0.1, 0.1))

        piece = Piece(given_coords)

        np.testing.assert_almost_equal(given_coords, piece.get_chop_coords())

    def test_fitness_perfect(self):
        given_coords = get_coords(10, 1.2, 0, 0.1)
        piece = Piece(given_coords)

        self.assertAlmostEqual(piece.get_fitness(), 0)

    def test_fitness_bad(self):
        given_coords = get_coords(10, 1.2, 0, 0.1)
        given_coords[1] -= 0.05
        given_coords[2] += 0.05
        given_coords[3] += 0.1
        piece = Piece(given_coords)

        self.assertGreater(piece.get_fitness(), 0)

    def test_fitness_worse(self):
        chop_coords = get_coords(10, 1.2, 0, 0.1)

        actual_coords_1 = np.copy(chop_coords)
        actual_coords_1[2] += 0.05
        piece_1 = Piece(actual_coords_1)

        actual_coords_2 = np.copy(chop_coords)
        actual_coords_2[2] += 0.05
        actual_coords_2[3] += 0.05
        piece_2 = Piece(actual_coords_2)

        self.assertGreater(piece_2.get_fitness(), piece_1.get_fitness())


class ApproximatorTests(unittest.TestCase):
    def test_is_simple(self):
        coords = np.linspace(-0.1, 0.1, num=20)

        self.assertTrue(Approximator(coords).is_simple)

    def test_is_not_simple(self):
        coords = np.linspace(-1, 1, num=10)
        coords[1] += 0.05

        self.assertFalse(Approximator(coords).is_simple)

    @parameterized.expand(
        (
            ([2, 7],),
            ([2, 4, 6, 8],),
            ([3, 5, 8],),
        )
    )
    def test_get_pieces(self, indexes):
        coords = get_coords(10, 1.2, 0.1, 0.1)
        appr = Approximator(coords)

        pieces = appr.get_pieces(indexes)
        self.assertEqual(len(pieces), len(indexes) + 1)

    def test_get_pieces_continuity(self):
        coords = get_coords(19, 1.2, 0.1, 0.1)
        appr = Approximator(coords)
        pieces = appr.get_pieces([5, 10, 15])

        piece_coords = []
        for piece in pieces:
            piece_coords += piece.coords.tolist()[:-1]
        piece_coords.append(pieces[-1].coords[-1])

        np.testing.assert_array_equal(coords, piece_coords)
