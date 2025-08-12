import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.base.exceptions import ClampExistsError
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.optimize.clamps.free import FreeClamp
from classy_blocks.optimize.clamps.surface import PlaneClamp
from classy_blocks.optimize.links import TranslationLink
from classy_blocks.optimize.optimizer import MeshOptimizer, SketchOptimizer
from classy_blocks.optimize.smoother import SketchSmoother
from classy_blocks.util import functions as f
from tests.test_optimize.optimize_fixtures import BoxTestsBase, SketchTestsBase


class MeshOptimizerTests(BoxTestsBase):
    @parameterized.expand(
        (
            (0.5, 0, 0.5),  # TODO: add intermediate iterations
            (1.0, 0, 1),  # relaxation disabled
        )
    )
    def test_relaxation(self, relaxation, iteration, result):
        optimizer = MeshOptimizer(self.mesh)
        optimizer.config.relaxation_start = relaxation
        self.assertAlmostEqual(optimizer.relaxation_factor(iteration), result)

    def test_add_junction_existing(self):
        optimizer = MeshOptimizer(self.mesh, report=False)
        optimizer.add_clamp(FreeClamp(self.mesh.vertices[0].position))

        with self.assertRaises(ClampExistsError):
            optimizer.add_clamp(FreeClamp(self.mesh.vertices[0].position))

    def test_optimize(self):
        # move a point, then optimize it back to
        # its initial-ish position
        vertex = self.get_vertex([0, 0, 0])
        vertex.move_to([0.3, 0.3, 0.3])

        optimizer = MeshOptimizer(self.mesh, report=False)

        clamp = FreeClamp(vertex.position)
        optimizer.add_clamp(clamp)
        optimizer.optimize(method="Powell", tolerance=0.1)

        np.testing.assert_almost_equal(vertex.position, [0, 0, 0], decimal=1)

    def test_optimize_linked(self):
        vertex = self.get_vertex([0, 0, 0])
        vertex.move_to([0.3, 0.3, 0.3])
        follower_vertex = next(iter(self.finder.find_in_sphere([0, 1, 0])))
        follower_vertex.move_to([0.3, 1.3, 0.3])

        link = TranslationLink(vertex.position, follower_vertex.position)
        clamp = FreeClamp(vertex.position)

        optimizer = MeshOptimizer(self.mesh, report=False)
        optimizer.add_clamp(clamp)
        optimizer.add_link(link)
        optimizer.optimize()  # method="L-BFGS-B")

        self.assertGreater(f.norm(follower_vertex.position - f.vector(0, 1, 0)), 0)
        np.testing.assert_almost_equal(vertex.position, [0, 0, 0], decimal=1)


class SketchOptimizerTests(SketchTestsBase):
    def test_optimize_manual(self):
        sketch = MappedSketch(self.positions, self.quads)
        clamp = PlaneClamp([1.2, 1.6, 0], [0, 0, 0], [0, 0, 1])

        optimizer = SketchOptimizer(sketch, report=False)
        optimizer.add_clamp(clamp)

        optimizer.optimize(method="L-BFGS-B")

        np.testing.assert_almost_equal(sketch.positions[4], [1, 1, 0], decimal=1)

    def test_optimize_auto(self):
        sketch = MappedSketch(self.positions, self.quads)

        optimizer = SketchOptimizer(sketch, report=False)
        optimizer.auto_optimize(method="L-BFGS-B")

        np.testing.assert_almost_equal(sketch.positions[4], [1, 1, 0], decimal=1)


class ComplexSketchTests(unittest.TestCase):
    """Tests on a real-life case"""

    # A degenerate starting configuration,
    # smoothed to just barely valid

    def setUp(self):
        positions = np.array(
            [
                [0.01672874, 0.02687117, 0.02099406],
                [0.01672874, 0.03602998, 0.02814971],
                [0.01371287, 0.04465496, 0.03488828],
                [0.00370317, 0.04878153, 0.0381123],
                [-0.00689365, 0.04569677, 0.03570223],
                [-0.01109499, 0.03743333, 0.02924612],
                [-0.00613247, 0.02943629, 0.02299815],
                [0.00472874, 0.02687117, 0.02099406],
                [0.00872874, 0.02687117, 0.02099406],
                [0.01272874, 0.02687117, 0.02099406],
                [0.0110113, 0.03091146, 0.02415068],
                [0.0110113, 0.03549087, 0.0277285],
                [0.00950337, 0.03980335, 0.03109779],
                [0.00449852, 0.04186664, 0.0327098],
                [-0.00079989, 0.04032426, 0.03150476],
                [-0.00290056, 0.03619254, 0.02827671],
                [-0.0004193, 0.03219402, 0.02515273],
                [0.0050113, 0.03091146, 0.02415068],
                [0.0070113, 0.03091146, 0.02415068],
                [0.0090113, 0.03091146, 0.02415068],
            ],
        )

        face_map = [
            # outer blocks
            [8, 9, 1, 0],  # 0
            [9, 10, 2, 1],  # 1
            [10, 11, 3, 2],  # 2
            [11, 12, 4, 3],  # 3
            [12, 13, 5, 4],  # 4
            [13, 14, 6, 5],  # 5
            [14, 15, 7, 6],  # 6
            # inner blocks
            [15, 14, 9, 8],  # 7
            [14, 13, 10, 9],  # 8
            [13, 12, 11, 10],  # 9
        ]

        self.sketch = MappedSketch(positions, face_map)

    def test_optimize(self):
        smoother = SketchSmoother(self.sketch)
        smoother.smooth()
        optimizer = SketchOptimizer(self.sketch, report=False)
        initial_quality = optimizer.grid.quality

        optimizer.auto_optimize(tolerance=1e-6, method="Nelder-Mead")

        self.assertLess(optimizer.grid.quality, initial_quality)
