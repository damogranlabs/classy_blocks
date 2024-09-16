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
            (0.035, 2),
            (0.01, 1),
            (1e-3, 1),
            (1e-4, 1),
        )
    )
    def test_get_from_length(self, length, chop_count):
        chops = self.params.get_chops_from_length(length)

        self.assertEqual(len(chops), chop_count)

    def test_length_ratio(self):
        chops = self.params.get_chops_from_length(1)

        for chop in chops:
            self.assertLessEqual(chop.length_ratio, 1)

    def test_continuity_laminar(self):
        """Last cell size from boundary layer must be smaller than buffer's start size"""
        chops = self.params.get_chops_from_length(1)

        boundary_data = chops[0].calculate(1)
        buffer_data = chops[1].calculate(1)

        self.assertGreaterEqual(buffer_data.start_size, boundary_data.end_size)

    def test_continuity_buffer(self):
        """Last cell size from boundary layer must be smaller than buffer's start size"""
        chops = self.params.get_chops_from_length(1)

        buffer_data = chops[1].calculate(1)
        bulk_data = chops[2].calculate(1)

        self.assertGreaterEqual(bulk_data.end_size, buffer_data.start_size)
