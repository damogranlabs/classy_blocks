import unittest

from parameterized import parameterized

from classy_blocks.grading.autochop.low_re import LowReChopParams


class LowReChopTests(unittest.TestCase):
    def setUp(self):
        self.params = LowReChopParams(1e-3, 0.01, 1.2, 30, 2, "avg")

    @parameterized.expand(
        (
            (2, 3),
            (1, 3),
            (0.04, 2),
            (0.01, 1),
            (1e-3, 1),
            (1e-4, 1),
        )
    )
    def test_get_from_length(self, length, chop_count):
        chops = self.params.get_chops_from_length(length)

        self.assertEqual(len(chops), chop_count)
