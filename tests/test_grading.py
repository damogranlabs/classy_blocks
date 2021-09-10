import unittest
import os

from classy_blocks.classes.grading import Grading

class TestGrading(unittest.TestCase):
    def setUp(self):
        self.g = Grading()

    def test_output_empty(self):
        self.assertEqual(str(self.g), '1')

    def test_output_single(self):
        self.g.add_division(1, 1, 3)
        self.assertEqual(str(self.g), '3')

    def test_output_multi(self):
        self.g.add_division(0.25, 0.4, 2)
        self.g.add_division(0.5,  0.2, 1)
        self.g.add_division(0.25, 0.4, 0.5)

        expected_output = "(" + os.linesep + \
	        "\t(0.25 0.4 2)" + os.linesep + \
	        "\t(0.5 0.2 1)" + os.linesep + \
	        "\t(0.25 0.4 0.5)" + os.linesep + \
            ")"

        self.assertEqual(str(self.g), expected_output)

    def test_copy(self):
        """ check that copy doesn't spoil the original """
        self.g.add_division(1, 1, 3)
        h = self.g.copy()

        h.expansion_ratios[0] = 5

        self.assertEqual(self.g.expansion_ratios[0], 3)

    def test_copy_invert_simple(self):
        self.g.add_division(1, 1, 5)
        h = self.g.copy(invert=True)

        self.assertEqual(self.g.expansion_ratios[0], 5)
        self.assertEqual(h.expansion_ratios[0], 0.2)