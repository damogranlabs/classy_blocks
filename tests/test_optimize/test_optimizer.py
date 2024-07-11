import numpy as np

from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.optimize.clamps.free import FreeClamp
from classy_blocks.optimize.clamps.surface import PlaneClamp
from classy_blocks.optimize.junction import ClampExistsError
from classy_blocks.optimize.links import TranslationLink
from classy_blocks.optimize.optimizer import MeshOptimizer, SketchOptimizer
from classy_blocks.util import functions as f
from tests.test_optimize.optimize_fixtures import BoxTestsBase, SketchTestsBase


class MeshOptimizerTests(BoxTestsBase):
    def test_add_junction_existing(self):
        optimizer = MeshOptimizer(self.mesh)
        optimizer.release_vertex(FreeClamp(self.mesh.vertices[0].position))

        with self.assertRaises(ClampExistsError):
            optimizer.release_vertex(FreeClamp(self.mesh.vertices[0].position))

    def test_optimize(self):
        # move a point, then optimize it back to
        # its initial-ish position
        vertex = self.get_vertex([0, 0, 0])
        vertex.move_to([0.3, 0.3, 0.3])

        optimizer = MeshOptimizer(self.mesh)

        clamp = FreeClamp(vertex.position)
        optimizer.release_vertex(clamp)
        optimizer.optimize()

        np.testing.assert_almost_equal(vertex.position, [0, 0, 0], decimal=1)

    def test_optimize_linked(self):
        vertex = self.get_vertex([0, 0, 0])
        vertex.move_to([0.3, 0.3, 0.3])
        follower_vertex = next(iter(self.finder.find_in_sphere([0, 1, 0])))

        link = TranslationLink(vertex.position, follower_vertex.position)
        clamp = FreeClamp(vertex.position)

        optimizer = MeshOptimizer(self.mesh)
        optimizer.release_vertex(clamp)
        optimizer.add_link(link)
        optimizer.optimize()

        self.assertGreater(f.norm(follower_vertex.position - f.vector(0, 1, 0)), 0)
        np.testing.assert_almost_equal(vertex.position, [0, 0, 0], decimal=1)


class SketchOptimizerTests(SketchTestsBase):
    def test_optimize_manual(self):
        sketch = MappedSketch(self.positions, self.quads)
        clamp = PlaneClamp([1.5, 1.5, 0], [0, 0, 0], [0, 0, 1])

        optimizer = SketchOptimizer(sketch)
        optimizer.release_vertex(clamp)

        optimizer.optimize(method="L-BFGS-B")

        np.testing.assert_almost_equal(sketch.positions[4], [1, 1, 0], decimal=3)

    def test_optimize_auto(self):
        sketch = MappedSketch(self.positions, self.quads)

        optimizer = SketchOptimizer(sketch)
        optimizer.auto_optimize(method="L-BFGS-B")

        np.testing.assert_almost_equal(sketch.positions[4], [1, 1, 0], decimal=3)
