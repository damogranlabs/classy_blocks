import unittest

import numpy as np

import classy_blocks as cb


class StackTests(unittest.TestCase):
    @property
    def base(self) -> cb.OneCoreDisk:
        return cb.OneCoreDisk([0, 0, 0], [1, 0, 0], [0, 0, 1])

    def test_construct_extruded(self):
        _ = cb.ExtrudedStack(self.base, 1, 4)

    def test_construct_revolved(self):
        _ = cb.RevolvedStack(self.base, np.pi / 6, [0, 1, 0], [2, 0, 0], 4)

    def test_construct_transformed(self):
        _ = cb.TransformedStack(
            self.base,
            [cb.Translation([0, 0, 1]), cb.Rotation([0, 0, 1], np.pi / 6, [0, 0, 0])],
            4,
            [cb.Translation([0, 0, 0.5]), cb.Rotation([0, 0, 1], np.pi / 12, [0, 0, 0])],
        )

    def test_chop(self):
        stack = cb.ExtrudedStack(self.base, 1, 4)

        stack.chop(count=10)

        for shape in stack.shapes:
            self.assertEqual(len(shape.grid[0][0].chops[2]), 1)
