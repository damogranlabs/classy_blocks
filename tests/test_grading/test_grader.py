import unittest

from classy_blocks.construct.operations.box import Box
from classy_blocks.grading.autograding.simple.grader import SimpleGrader
from classy_blocks.mesh import Mesh


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()
        self.mesh.add(Box([0, 0, 0], [1, 1, 1]))

    def test_at_wall(self):
        self.mesh.set_default_patch("sides", "wall")

        grader = SimpleGrader(self.mesh, 0.05)
        at_wall = grader.check_at_wall(grader.probe.get_rows(0)[0])

        self.assertTrue(at_wall[0])
        self.assertTrue(at_wall[1])

    def test_no_set_count(self):
        self.mesh.operations[0].chop(0, count=3)

        grader = SimpleGrader(self.mesh, 0.05)
        grader.grade()

        # make sure manually set counts are respected
        self.assertEqual(self.mesh.blocks[0].axes[0].count, 3)
