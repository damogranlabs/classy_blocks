import numpy as np

from classy_blocks.optimize.smoother import MeshSmoother, SmootherBase
from tests.test_optimize.optimize_fixtures import BoxTestsBase


class SmootherTests(BoxTestsBase):

    def test_smooth(self):
        vertex = self.get_vertex([0, 0, 0])
        vertex.move_to([0.3, 0.3, 0.3])

        grid = self.get_grid(self.mesh)

        np.testing.assert_almost_equal(grid.points[vertex.index], [0.3, 0.3, 0.3])

        smoother = SmootherBase(grid)
        smoother.smooth()

        np.testing.assert_almost_equal(grid.points[vertex.index], [0, 0, 0])

    def test_smooth_mesh(self):
        vertex = self.get_vertex([0, 0, 0])
        vertex.move_to([0.3, 0.3, 0.3])

        smoother = MeshSmoother(self.mesh)
        smoother.smooth()

        np.testing.assert_almost_equal(self.mesh.vertices[vertex.index].position, [0, 0, 0])
