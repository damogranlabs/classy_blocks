import numpy as np

from classy_blocks.construct.flat.sketches.disk import OneCoreDisk
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.optimize.smoother import MeshSmoother, SketchSmoother, SmootherBase
from classy_blocks.util import functions as f
from tests.test_optimize.optimize_fixtures import BoxTestsBase, SketchTestsBase


class HexSmootherTests(BoxTestsBase):

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


class QuadSmootherTests(SketchTestsBase):
    def test_smooth_disk(self):
        sketch = OneCoreDisk([0, 0, 0], [1, 0, 0], [0, 0, 1])
        smoother = SketchSmoother(sketch)

        # laplacian smoothing produces inner square that is much smaller than it should be;
        # therefore it is handled manually
        radius_pre = f.norm(sketch.positions[0])
        smoother.smooth()
        radius_post = f.norm(sketch.positions[0])

        self.assertLess(radius_post, radius_pre)

    def test_smooth(self):
        # a grid of vertices 3x3
        sketch = MappedSketch(self.positions, self.quads)
        smoother = SketchSmoother(sketch)

        smoother.smooth()

        np.testing.assert_almost_equal(sketch.positions[4], [1, 1, 0], decimal=5)
