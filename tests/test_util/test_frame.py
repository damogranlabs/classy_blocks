import unittest

from classy_blocks.util.frame import Frame


class FrameTests(unittest.TestCase):
    def test_add_beam_invalid(self):
        frame = Frame()

        with self.assertRaises(ValueError):
            frame.add_beam(0, 0, 1)
