import unittest

import numpy as np
from parameterized import parameterized

import classy_blocks as cb


class StackTests(unittest.TestCase):
    @property
    def round_base(self) -> cb.OneCoreDisk:
        return cb.OneCoreDisk([0, 0, 0], [1, 0, 0], [0, 0, 1])

    @property
    def square_base(self) -> cb.Grid:
        return cb.Grid([0, 0, 0], [1, 1, 0], 2, 5)

    def test_construct_extruded(self):
        stack = cb.ExtrudedStack(self.round_base, 1, 4)

        self.assertEqual(len(stack.grid), 4)

    def test_construct_revolved(self):
        _ = cb.RevolvedStack(self.round_base, np.pi / 6, [0, 1, 0], [2, 0, 0], 4)

    def test_construct_transformed(self):
        _ = cb.TransformedStack(
            self.round_base,
            [cb.Translation([0, 0, 1]), cb.Rotation([0, 0, 1], np.pi / 6, [0, 0, 0])],
            4,
            [cb.Translation([0, 0, 0.5]), cb.Rotation([0, 0, 1], np.pi / 12, [0, 0, 0])],
        )

    def test_chop(self):
        stack = cb.ExtrudedStack(self.round_base, 1, 4)

        stack.chop(count=10)

        for shape in stack.shapes:
            self.assertEqual(len(shape.grid[0][0].chops[2]), 1)

    def test_grid_square_axis0(self):
        stack = cb.ExtrudedStack(self.square_base, 1, 3)

        self.assertEqual(len(stack.grid), 3)

    def test_grid_square_axis1(self):
        stack = cb.ExtrudedStack(self.square_base, 1, 3)

        self.assertEqual(len(stack.grid[0]), 5)

    def test_grid_square_axis2(self):
        stack = cb.ExtrudedStack(self.square_base, 1, 5)

        self.assertEqual(len(stack.grid[0][0]), 2)

    def test_grid_round_axis0(self):
        stack = cb.ExtrudedStack(self.round_base, 1, 3)

        self.assertEqual(len(stack.grid), 3)

    def test_grid_round_axis1(self):
        stack = cb.ExtrudedStack(self.round_base, 1, 3)

        self.assertEqual(len(stack.grid[0]), 2)

    def test_grid_round_axis2(self):
        stack = cb.ExtrudedStack(self.round_base, 1, 3)

        self.assertEqual(len(stack.grid[0][0]), 1)  # core
        self.assertEqual(len(stack.grid[0][1]), 4)  # shell

    @parameterized.expand(
        (
            # x-axis
            (0, 0, 15),
            (0, 1, 15),
            # y-axis
            (1, 0, 6),
            (1, 1, 6),
            (1, 2, 6),
            (1, 3, 6),
            (1, 4, 6),
            # z-axis: stacked shapes
            (2, 0, 10),
            (2, 1, 10),
            (2, 2, 10),
        )
    )
    def test_get_slice_square(self, axis, index, count):
        stack = cb.ExtrudedStack(self.square_base, 1, 3)

        self.assertEqual(len(stack.get_slice(axis, index)), count)
